import logging
import multiprocessing as mp
import signal
import sys
import time
from ctypes import c_longlong
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aubio  # type: ignore
import numpy as np  # type: ignore
import pyaudio  # type: ignore

from . import bandpass, buffer, dsp
from .child_process import Process

LOGGER = logging.getLogger(__name__)


class PyaudioCallback:
    def __init__(self, buffer_queue: mp.Queue):
        self.buffer_id = 0
        self._input_buffer_queue = buffer_queue

    def __call__(self, in_data, frame_count, time_info, flag):  # pylint: disable=W0613
        LOGGER.debug(f"buffer {self.buffer_id} recieved")
        timestamp = time.perf_counter()
        self._input_buffer_queue.put((timestamp, self.buffer_id, in_data))
        self.buffer_id += 1

        if not mp.parent_process().is_alive():
            LOGGER.critical("pyaudio callback orphaned, killing")
            sys.exit()

        return (None, pyaudio.paContinue)


class PyMVF:
    def __init__(
        self,
        bin_edges: List[int],
        sample_rate: int = 44100,
        buffer_size: int = 512,
        buffer_discard_qty: int = int(44100 / 512),
        output_queue: Optional[mp.Queue] = None,
    ):
        LOGGER.info("Initializing PyMVF")
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        if output_queue is not None:
            self.output_queue = output_queue
        else:
            self.output_queue = mp.Queue()

        self._bin_edges = bin_edges
        self._buffer_discard_qty = buffer_discard_qty

        self._input_buffer_queue: mp.Queue = mp.Queue()

        self._input_stream_process = Process(target=self._input_stream)
        self._input_stream_process.start()
        LOGGER.info("Initialized Stream")

        self._main_process = Process(target=self._main)
        self._main_process.start()
        LOGGER.info("Initialized PyMVF")

    def _main(self) -> None:
        detect_beat = aubio.tempo(
            method="default",
            buf_size=self.buffer_size * 2,
            hop_size=self.buffer_size,
            samplerate=self.sample_rate,
        )

        buffer_stub_counter = 0
        processed_buffer_counter = 0

        left_calculate_bin_rms = bandpass.CalculateBinRMS(
            self.sample_rate, self.buffer_size, self._bin_edges, 12
        )
        right_calculate_bin_rms = bandpass.CalculateBinRMS(
            self.sample_rate, self.buffer_size, self._bin_edges, 12
        )

        channel_arrays: Dict[str, np.ndarray] = {}
        buffer_stubs = []

        while self._buffer_discard_qty:
            _, buffer_id, _ = self._input_buffer_queue.get()
            if buffer_id + 1 == self._buffer_discard_qty:
                break

        while True:
            timestamp, buffer_id, stereo_bytes = self._input_buffer_queue.get()
            recieved_buffer_counter = (buffer_id - self._buffer_discard_qty) + 1
            # pyaudio does not give good date for the first several buffers
            # a conservative amount of delay is introduced here

            LOGGER.debug(f"Processing buffer {buffer_id}")
            stereo_array = np.frombuffer(stereo_bytes, dtype=np.float32)

            left_channel_array, right_channel_array = dsp.split_channels(stereo_array)

            mono_array = np.add(left_channel_array / 2, right_channel_array / 2)

            # we do not have all the info for a full buffer object until we bandpass
            buffer_stub = buffer.create_buffer_stub(
                buffer_id,
                timestamp,
                mono_array,
                left_channel_array,
                right_channel_array,
                detect_beat,
            )
            buffer_stubs.append(buffer_stub)
            buffer_stub_counter += 1
            assert buffer_stub_counter == recieved_buffer_counter

            # handle bandpass filter stuff
            if channel_arrays.get("left") is None:
                channel_arrays["left"] = left_channel_array
                channel_arrays["right"] = right_channel_array
            else:
                channel_arrays["left"] = np.concatenate(
                    (channel_arrays["left"], left_channel_array)
                )
                channel_arrays["right"] = np.concatenate(
                    (channel_arrays["right"], right_channel_array)
                )
            LOGGER.debug(f"Buffer stub for {buffer_id} processed")

            if len(channel_arrays["left"]) < self.sample_rate:
                continue

            LOGGER.debug("Filtering buffers")
            if len(channel_arrays["left"]) == self.sample_rate:
                LOGGER.debug("Length of channel arrays equal to `sample_rate`")

                left_bin_energies_mapping_list = left_calculate_bin_rms(
                    channel_arrays["left"]
                )
                right_bin_energies_mapping_list = right_calculate_bin_rms(
                    channel_arrays["right"]
                )
                del channel_arrays["left"]
                del channel_arrays["right"]
            else:
                # just send `sample_rate`s (1 second) worth of samples
                left_bin_energies_mapping_list = left_calculate_bin_rms(
                    channel_arrays["left"][: self.sample_rate]
                )
                right_bin_energies_mapping_list = right_calculate_bin_rms(
                    channel_arrays["right"][: self.sample_rate]
                )
                # remove all samples besides the remainder
                channel_arrays["left"] = channel_arrays["left"][self.sample_rate :]
                channel_arrays["right"] = channel_arrays["right"][self.sample_rate :]

            # averaging the energies of each channel gives us the mono channel's energy
            mono_bin_energies_mapping_list = []
            for left_bin_energy_mapping, right_bin_energy_mapping in zip(
                left_bin_energies_mapping_list, right_bin_energies_mapping_list
            ):
                mono_bin_energies_mapping = {}
                for (bin_, left_energy), right_energy in zip(
                    left_bin_energy_mapping.items(), right_bin_energy_mapping.values()
                ):
                    mono_bin_energies_mapping[bin_] = (left_energy + right_energy) / 2
                mono_bin_energies_mapping_list.append(mono_bin_energies_mapping)

            assert (
                len(mono_bin_energies_mapping_list)
                == len(left_bin_energies_mapping_list)
                == len(right_bin_energies_mapping_list)
            )

            assert (
                len(mono_bin_energies_mapping_list) + 1
                >= buffer_stub_counter - processed_buffer_counter
            )

            # we do not nessesarilly get as many energies as we filtered
            # this is compensated for in the filterbank process by sometimes sending
            #    more than we sent, but we have to compensate for that here as well
            stubs_to_promote = buffer_stubs[: len(left_bin_energies_mapping_list)]
            buffer_stubs = buffer_stubs[len(left_bin_energies_mapping_list) :]

            assert len(stubs_to_promote) == len(mono_bin_energies_mapping_list)

            # weave together all the buffer stubs and bin energy mappings
            for (
                stub,
                mono_bin_energy_mapping,
                left_bin_energy_mapping,
                right_bin_energy_mapping,
            ) in zip(
                stubs_to_promote,
                mono_bin_energies_mapping_list,
                left_bin_energies_mapping_list,
                right_bin_energies_mapping_list,
            ):
                finished_buffer = buffer.Buffer(
                    id=stub.id,
                    timestamp=stub.timestamp,
                    mono_rms=stub.mono_rms,
                    left_rms=stub.left_rms,
                    right_rms=stub.right_rms,
                    beat=stub.beat,
                    mono_bin_energies=mono_bin_energy_mapping,
                    left_bin_energies=left_bin_energy_mapping,
                    right_bin_energies=right_bin_energy_mapping,
                )
                self.output_queue.put(finished_buffer)
                processed_buffer_counter += 1

            LOGGER.info(
                f"Processed:{processed_buffer_counter} "
                f"Stubs:{buffer_stub_counter} "
                f"Recieved:{recieved_buffer_counter}"
            )

    def __call__(self) -> buffer.Buffer:
        """ Return one buffer from the output queue

        Blocks until a buffer is processed.
        Use this as an alternative to accessing `output_queue` directly.

        Returns:
            Buffer: a processed buffer.
        """
        processed_buffer = self.output_queue.get()

        return processed_buffer

    def _input_stream(self) -> None:
        """ input stream method ran in a child process"""
        callback = PyaudioCallback(self._input_buffer_queue)
        pyaudio.PyAudio().open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=callback,
        )
        signal.pause()  # type:ignore
