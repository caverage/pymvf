import logging
import multiprocessing as mp
import signal
import time
from ctypes import c_longlong
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import pyaudio  # type: ignore

from . import buffer, signal_processing
from .child_process import Process

LOGGER = logging.getLogger(__name__)


class PyaudioCallback:
    def __init__(self, buffer_queue: mp.Queue):
        self.buffer_id = 0
        self._buffer_queue = buffer_queue

    def __call__(self, in_data, frame_count, time_info, flag):  # pylint: disable=W0613
        LOGGER.info(f"buffer {self.buffer_id} recieved")
        timestamp = time.perf_counter()
        self._buffer_queue.put((timestamp, self.buffer_id, in_data))
        self.buffer_id += 1
        return (None, pyaudio.paContinue)


class PyMVF:
    def __init__(
        self, bin_edges: List[int], output_queue: mp.Queue, buffer_discard_qty: int = 10
    ):
        LOGGER.info("Initializing PyMVF")
        self.sample_rate = 44100
        self.buffer_size = 512

        self._buffer_queue: mp.Queue = mp.Queue()
        self._error_queue: mp.Queue = mp.Queue()

        self._left_calculate_bin_rms = signal_processing.CalculateBinRMS(
            self.sample_rate, bin_edges, 6
        )
        self._right_calculate_bin_rms = signal_processing.CalculateBinRMS(
            self.sample_rate, bin_edges, 6
        )

        self._input_stream_process = Process(target=self._input_stream)
        self._input_stream_process.start()
        self.stream = None
        LOGGER.info("Initialized Stream")

        # self._filterbank = signal_processing.FilterBank(
        #     self.sample_rate, self.buffer_size
        # )

        while True:
            timestamp, buffer_id, stereo_buffer = self._buffer_queue.get()
            # pyaudio does not give good date for the first several buffers
            # a conservative amount of delay is introduced here
            if buffer_id < buffer_discard_qty:
                continue

            LOGGER.info(f"Processing buffer {buffer_id}")
            output_queue.put(
                buffer.create_buffer(
                    buffer_id,
                    timestamp,
                    stereo_buffer,
                    self._left_calculate_bin_rms,
                    self._right_calculate_bin_rms,
                )
            )
            LOGGER.info(f"Buffer {buffer_id} processed")

    def _input_stream(self) -> None:
        """ input stream method ran in a child process"""
        callback = PyaudioCallback(self._buffer_queue)
        self.stream = pyaudio.PyAudio().open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=callback,
        )
        signal.pause()  # type:ignore
