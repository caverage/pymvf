""" Module for signal processing helpers and utilities"""

import logging
import multiprocessing as mp
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


def get_rms(samples: np.array) -> float:
    """ Get the RMS of an array of audio samples

    Args:
        samples: the samples to get the RMS from

    Returns:
        float: the RMS
    """
    return np.sqrt(np.mean(samples ** 2))  # type:ignore

    # return np.sqrt(sum(samples * samples) / len(samples))  # type:ignore


class _BandpassFilter:
    """ A callable singleton implementing a Butterworth bandpass filter.

    Args:
        low_cut: Hz to attenuate all frequencies below
        high_cut: Hz to attenuate all frequencies above
        sample_rate: sample rate that samples to the filter are input at
        order: band pass filter order
    """

    def __init__(
        self,
        shared_buffer_name: str,
        sample_rate: int,
        buffer_size: int,
        low_cut: int,
        high_cut: int,
        order: int,
    ):
        self.name = f"{low_cut}-{high_cut} filter"

        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self.shared_buffer_name = shared_buffer_name

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
        shared_buffer_memory = shared_memory.SharedMemory(name=self.shared_buffer_name)
        buffer = np.ndarray(
            (self._buffer_size,), dtype=np.float32, buffer=shared_buffer_memory.buf
        )

        filter_state = scipy.signal.sosfilt_zi(sos)
        previous_buffer = None
        while True:
            # block until called
            self._input_queue.get()
            LOGGER.debug(f"{self.name} recieved block")
            if previous_buffer is None:
                previous_buffer = buffer

            # buffering the input block achieves a steady state in the filter
            buffered_buffer = np.append(previous_buffer, buffer)

            filtered_buffered_buffer, filter_state = scipy.signal.sosfilt(
                sos, buffered_buffer, zi=filter_state
            )
            # cut off the transient buffer, leaving the filtered, steady state pass band
            filtered_buffer = filtered_buffered_buffer[len(previous_buffer) :]

            previous_buffer = buffer
            self._output_queue.put(filtered_buffer)
            LOGGER.debug(f"{self.name} processed block")

    @property
    def result(self) -> np.array:
        """ Get result of most recent call.

        Returns:
            np.array: filtered block
        """
        return self._output_queue.get()

    def __call__(self) -> None:
        """ filter a given block of audio samples

        An effort is made to reduce (eliminate?) transients present at the begining
            of the filter.

        See: dsprelated.com/freebooks/filters/Transient_Response_Steady_State.html
        See: https://dsp.stackexchange.com/questions/70940/
        """

        self._input_queue.put(True)


class CalculateBinRMS:
    """ A callable singleton for getting the RMS of each bin.

    Args:
        sample_rate: sample rate that samples to the filter are input at
        order: band pass filter order, should be odd, between 3 and 9
    """

    def __init__(
        self, sample_rate: int, buffer_size: int, bin_edges: List[int], order: int
    ):
        self.shared_buffer_memory = shared_memory.SharedMemory(
            create=True, size=buffer_size * 4
        )
        self.shared_buffer = np.ndarray(
            (buffer_size,), dtype=np.float32, buffer=self.shared_buffer_memory.buf
        )

        self._bandpass_filters: Dict[Tuple[int, int], _BandpassFilter] = {}
        previous_edge = bin_edges[0]
        for edge in bin_edges[1:]:
            self._bandpass_filters[(previous_edge, edge)] = _BandpassFilter(
                self.shared_buffer_memory.name,
                sample_rate,
                buffer_size,
                previous_edge,
                edge,
                order,
            )
            previous_edge = edge

        LOGGER.debug("CalculateBinRMS created")

    def __call__(self, samples: np.ndarray) -> Dict[Tuple[int, int], float]:
        """ get list of rms values for each bin

        Args:
            samples: input to be processed

        Returns:
            List[float]: rms values for each bin
        """
        self.shared_buffer[:] = samples[:]
        LOGGER.info(samples.dtype)
        LOGGER.info(self.shared_buffer.dtype)

        # tell children to get to work
        for bandpass_filter in self._bandpass_filters.values():
            bandpass_filter()
        LOGGER.debug("all filters processing")

        # collect the results into a dict
        amplitudes = {}
        for bin_, bandpass_filter in self._bandpass_filters.items():
            amplitudes[bin_] = float(get_rms(bandpass_filter.result))
        LOGGER.debug("all results recieved")

        return amplitudes


# class FilterBank:
#     def __init__(self, sample_rate: int, buffer_size: int) -> None:
#         self._fft = aubio.fft(buffer_size)
#         self._filterbank = aubio.filterbank(len(BINS) - 2, buffer_size)
#         self._filterbank.set_power(3)
#         self._filterbank.set_triangle_bands(aubio.fvec(BINS), sample_rate)
#
#         coefficients = self._filterbank.get_coeffs()
#
#         # increase the relative power of the higher bins
#         for i, coefficient in enumerate(coefficients):
#             # first 2 coefficients stay the same
#             coefficients[i] = coefficient * (i ** 2.8)
#
#         self._filterbank.set_coeffs(coefficients)
#
#     def __call__(self, input_array: np.ndarray) -> np.ndarray:
#         fft = self._fft(input_array)
#         # we don't need much precision at all
#         energy = self._filterbank(fft).astype(np.uint32)
#
#         return energy
