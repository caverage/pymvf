""" Module for signal processing helpers and utilities"""

import logging
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from typing import Dict, List, Tuple

import numpy as np  # type:ignore
import scipy.signal  # type:ignore

from .child_process import Process

LOGGER = logging.getLogger(__name__)


def erb_from_freq(freq: int) -> float:
    """ Get equivalent rectangular bandwidth from the given frequency.

    See: https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth

    Args:
        freq: input frequency

    Returns:
        int: cam
    """

    return float(9.265 * np.log(1 + np.divide(freq, 24.7 * 9.16)))


def generate_bin_edges(low_freq: int, high_freq: int, count: int) -> List[int]:
    """ Bin sizes designed around Equivalent Rectangular Bandwidth

    NOTE: this becomes less accurate as the bin values increase,
        but should be good enough (TM)

    Args:
        low_freq: where to start the bins (> 0)
        high_freq: where to end the bins (<= 20,000)
        count: number of bin edges to generate

    Returns:
        List[int]: bin seperators in Hz
    """

    bin_edges = []
    cams = np.linspace(erb_from_freq(low_freq), erb_from_freq(high_freq), count)
    for i, cam in enumerate(cams):
        if not i:
            # this is probably not nessesary, but better safe than sorry?
            bin_edges.append(low_freq)
        elif i == len(cams) - 1:
            bin_edges.append(high_freq)
        else:
            bin_edges.append(round(10 ** (cam / 21.4) / 0.00437 - 1 / 0.00437))

    return bin_edges


def get_rms(sample_block: np.array) -> float:
    """ Get the RMS of an array of audio sample_block

    Args:
        sample_block: the sample_block to get the RMS from

    Returns:
        float: the RMS
    """
    return np.sqrt(np.mean(sample_block ** 2))  # type:ignore

    # return np.sqrt(sum(sample_block * sample_block) / len(sample_block))  # type:ignore


class _BandpassFilter:
    """ A callable singleton implementing a Butterworth bandpass filter.

    Args:
        low_cut: Hz to attenuate all frequencies below
        high_cut: Hz to attenuate all frequencies above
        sample_rate: sample rate that sample_block to the filter are input at
        order: band pass filter order
    """

    def __init__(
        self,
        shared_sample_block_name: str,
        sample_rate: int,
        buffer_size: int,
        low_cut: int,
        high_cut: int,
        order: int,
    ):
        self.name = f"{low_cut}-{high_cut} filter"

        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._shared_sample_block_name = shared_sample_block_name

        self._remainder = None

        self._input_queue: mp.Queue = mp.Queue()
        self._output_queue: mp.Queue = mp.Queue()

        sos = self._design_filter(low_cut, high_cut, sample_rate, order)
        self._child_process_object = Process(
            target=self._child_process, name=self.name, args=(sos,)
        )

        self._child_process_object.start()
        LOGGER.info(f"started {self.name}")

    @staticmethod
    def _design_filter(
        low_cut: int, high_cut: int, sample_rate: int, order: int
    ) -> np.ndarray:
        """ Get the initial filter state

        Args:
            low_cut: See `__init__`
            high_cut: See `__init__`
            sample_rate: See `__init__`
            order: See `__init__`

        Returns:
            np.ndarray: array of second order filter coefficients
        """

        nyq = 0.5 * sample_rate
        low = low_cut / nyq
        high = high_cut / nyq
        return scipy.signal.butter(order, [low, high], btype="band", output="sos")

    def _child_process(self, sos: np.ndarray) -> None:
        shared_sample_block_memory = shared_memory.SharedMemory(
            name=self._shared_sample_block_name
        )
        sample_block = np.ndarray(
            (self._sample_rate * 2,),
            dtype=np.float32,
            buffer=shared_sample_block_memory.buf,
        )

        filter_state = scipy.signal.sosfilt_zi(sos)
        while True:
            # block until called
            self._input_queue.get()
            start_time = time.monotonic()

            filtered_sample_block, filter_state = scipy.signal.sosfilt(
                sos, sample_block, zi=filter_state
            )
            # cut off the transient buffer, leaving the filtered, steady state pass band
            filtered_sample_block = filtered_sample_block[
                self._sample_rate - self._buffer_size :
            ]

            self._output_queue.put(filtered_sample_block)
            LOGGER.debug(
                f"filter_{self.name}: processed block in {round(time.monotonic()-start_time,5)}"
            )

    @property
    def result(self) -> np.array:
        """ Get result of most recent call.

        Returns:
            np.array: filtered block
        """
        result = self._output_queue.get()

        if self._remainder is not None:
            result = np.concatenate((self._remainder, result))

        remainder_len = len(result) % self._buffer_size

        if remainder_len:
            self._remainder = result[-remainder_len:]
        else:
            self._remainder = None

        result = result[:-remainder_len]

        return result

    def __call__(self) -> None:
        """ filter a given block of audio sample_block

        See: dsprelated.com/freebooks/filters/Transient_Response_Steady_State.html
        See: https://dsp.stackexchange.com/questions/70940/
        """

        self._input_queue.put(True)


class _BufferSampleBlock:
    def __init__(self, sample_rate: int, buffer_size: int):
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size

        self._previous_sample_block = None
        self._remainder = None

    def __call__(self, sample_block: np.ndarray) -> np.ndarray:
        assert (
            len(sample_block) == self._sample_rate
        ), f"length {len(sample_block)} not equal to sample rate: {self._sample_rate}"

        if self._previous_sample_block is None:
            # prime _previous_sample_block with a duplicate to prevent initial transient
            self._previous_sample_block = sample_block

        if self._remainder is not None:
            previous_remainder_len = len(self._remainder)
            input_sample_block = np.concatenate((self._remainder, sample_block))
            remainder_len = len(input_sample_block) % self._buffer_size
        else:
            previous_remainder_len = 0
            input_sample_block = sample_block
            remainder_len = len(input_sample_block) % self._buffer_size

        if remainder_len:
            self._remainder = input_sample_block[-remainder_len:]
            input_sample_block = input_sample_block[:-remainder_len]
        else:
            # trying to slice with `0` returns the entire array :face_palm:
            self._remainder = None

        # add a transient prevention buffer to the input sample block
        # the legth that this buffer needs to be is likely way smaller than this,
        #    but the author is big dumb, so this is what we do
        # buffering the input block achieves a steady state in the filter
        if previous_remainder_len:
            input_sample_block = np.concatenate(
                (
                    self._previous_sample_block[
                        self._buffer_size
                        - previous_remainder_len : -previous_remainder_len
                    ],
                    input_sample_block,
                )
            )
        else:
            input_sample_block = np.concatenate(
                (self._previous_sample_block[self._buffer_size :], input_sample_block,)
            )

        self._previous_sample_block = sample_block
        return input_sample_block


class CalculateBinRMS:
    """ A callable singleton for getting the RMS of each bin.

    The output of this is delayed by (at least) u1 second.

    Args:
        sample_rate: sample rate that sample_block to the filter are input at
        order: band pass filter order, should be odd, between 3 and 9
    """

    def __init__(
        self, sample_rate: int, buffer_size: int, bin_edges: List[int], order: int
    ):
        self._buffer_size = buffer_size

        # sample block with room for 2 blocks of 4 bytes per sample
        self.shared_sample_block_memory = shared_memory.SharedMemory(
            create=True, size=(sample_rate * 4 * 2)
        )

        # create all bandpass filters
        self._bandpass_filters: Dict[Tuple[int, int], _BandpassFilter] = {}
        previous_edge = bin_edges[0]
        for edge in bin_edges[1:]:
            self._bandpass_filters[(previous_edge, edge)] = _BandpassFilter(
                self.shared_sample_block_memory.name,
                sample_rate,
                buffer_size,
                previous_edge,
                edge,
                order,
            )
            previous_edge = edge

        self._buffer_sample_block = _BufferSampleBlock(sample_rate, buffer_size)
        # ironic
        self._remainder = None

    def __call__(self, sample_block: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        """ get list of rms values for each bin

        An effort is made to reduce (eliminate?) transients present at the begining
            of the filter by prepending the previous input sample block to the one we
            care about.

        Args:
            sample_block: input to be processed

        Returns:
            Dict[Tuple[int, int], float]: rms values for each bin
        """

        buffered_sample_block = self._buffer_sample_block(sample_block)

        self.shared_sample_block = np.ndarray(
            (len(buffered_sample_block),),
            dtype=np.float32,
            buffer=self.shared_sample_block_memory.buf,
        )

        self.shared_sample_block[:] = buffered_sample_block[:]

        # # tell children to get to work
        for bandpass_filter in self._bandpass_filters.values():
            bandpass_filter()
        LOGGER.debug("all filters processing")

        # collect the results into a dict
        amplitudes = {}
        for bin_, bandpass_filter in self._bandpass_filters.items():
            energy_list = []
            filtered_sample_block = bandpass_filter.result
            for buffer in filtered_sample_block.reshape(
                int(len(filtered_sample_block) / self._buffer_size), self._buffer_size
            ):
                energy_list.append(float(get_rms(buffer)))
            amplitudes[bin_] = np.array(energy_list)
        LOGGER.debug("all results recieved")

        return amplitudes
