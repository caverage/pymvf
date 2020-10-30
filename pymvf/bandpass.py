""" bandpass filter related objects"""

import logging
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np  # type:ignore
import scipy.signal  # type:ignore

from pymvf import dsp

from .child_process import Process

LOGGER = logging.getLogger(__name__)


class MaxEnergyTracker:
    def __init__(self, buffers_per_second: float):
        self._buffers_per_second = buffers_per_second
        self.max_energy: Optional[float] = None
        self.all_time_max_energy: Optional[float] = None

    def __call__(self, energy: float) -> Optional[float]:
        if not energy:
            # don't try to divide by 0
            return self.max_energy

        if self.max_energy is None:
            self.max_energy = energy
            self.all_time_max_energy = energy
            return self.max_energy

        assert isinstance(self.all_time_max_energy, float)
        if energy > self.max_energy:
            self.max_energy = energy
            # LOGGER.info(f"max increased to {self.max_energy}")
            if self.max_energy > self.all_time_max_energy:
                self.all_time_max_energy = self.max_energy
            return self.max_energy

        # don't derease lower than 1/3 the total maximum
        if self.max_energy / self.all_time_max_energy < 0.333:
            return self.max_energy

        decrease_ammount = self.all_time_max_energy / (self._buffers_per_second * 10)
        # decrease the max energy by 1/3th of the all time max energy per second
        LOGGER.info(
            f"max decreased by {round(decrease_ammount/self.max_energy, 2)} percent"
        )

        self.max_energy = self.max_energy - decrease_ammount

        return self.max_energy


class BinsIntensity:
    def __init__(self, bins: List[Tuple[int, int]]):
        self.max_energy_trackers = [MaxEnergyTracker(bin_) for bin_ in bins]

    def __call__(self, bin_intensity_array: np.array) -> np.array:
        """ get array of bin intensities adjusted by their max energy size per bin

        Args:
            bin_intensity_array: sorted array of bin intensities

        Returns:
            np.array: array of adjusted intensity between 0 and 1

        """
        output_bin_adjusted_intensity: List[float] = []

        for max_energy_tracker, energy in zip(
            self.max_energy_trackers, bin_intensity_array
        ):
            output_bin_adjusted_intensity.append(energy / max_energy_tracker(energy))

        return np.array(output_bin_adjusted_intensity)


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

        self.max_energy_tracker = MaxEnergyTracker(
            self._sample_rate / self._buffer_size
        )

        self._remainder = None

        self._input_queue: mp.Queue = mp.Queue()
        self._output_queue: mp.Queue = mp.Queue()

        sos = self._design_filter(low_cut, high_cut, sample_rate, order)

        self._child_process_object = Process(
            target=self._child_process, name=self.name, args=(sos,)
        )
        self._child_process_object.start()

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
            filtered_sample_block = filtered_sample_block[self._sample_rate :]

            assert len(filtered_sample_block) == self._sample_rate

            self._output_queue.put(filtered_sample_block)

    @property
    def result(self) -> np.array:
        """ Get result of most recent call.

        Will returns a number of samples evenly divisible by the buffer_size, and hold
            the remainder for the next call.

        Returns:
            np.array: filtered block evenly divisible by buffer_size
        """
        result = self._output_queue.get()

        if self._remainder is not None:
            result = np.concatenate((self._remainder, result))

        remainder_len = len(result) % self._buffer_size

        if remainder_len:
            self._remainder = result[-remainder_len:]
            result = result[:-remainder_len]
        else:
            self._remainder = None

        return result

    def __call__(self) -> None:
        """ filter a given block of audio sample_block

        See: dsprelated.com/freebooks/filters/Transient_Response_Steady_State.html
        See: https://dsp.stackexchange.com/questions/70940/
        """

        self._input_queue.put(True)


class _BufferSampleBlock:
    """ Buffer a sample block and return a sample block that is divisible evenly

    How this works is not fully understood, but it is probobaly not perfect

    Args:
        sample_rate: the sample rate
        buffer_size: the buffer size
    """

    def __init__(self, sample_rate: int, buffer_size: int):
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size

        self._previous_sample_block = None
        # self._remainder = None

    def __call__(self, sample_block: np.ndarray) -> np.ndarray:
        """ See _BufferSampleBlock.__doc__

        Args:
            sample_block: the sample block to be buffered

        Returns:
            np.ndarray: a buffered sample block that is evenly divisible by buffer_size
        """
        assert (
            len(sample_block) == self._sample_rate
        ), f"length {len(sample_block)} not equal to sample rate: {self._sample_rate}"

        if self._previous_sample_block is None:
            # prime _previous_sample_block with a duplicate to prevent initial transient
            self._previous_sample_block = sample_block

        # add a transient prevention buffer to the input sample block
        # the legth that this buffer needs to be is likely way smaller than this,
        #    but the author is big dumb, so this is what we do
        # buffering the input block achieves a steady state in the filter
        buffered_sample_block = np.concatenate(
            (self._previous_sample_block, sample_block,)
        )

        self._previous_sample_block = sample_block

        assert len(buffered_sample_block) == self._sample_rate * 2
        return buffered_sample_block


class CalculateBinRMS:
    """ Apply a bandpass filter to a number of frequency bins.

    The output of this is delayed by (at least) 1 second.

    Args:
        sample_rate: sample rate that sample_block to the filter are input at
        buffer_size: the buffer size
        bin_edges: the edges used to create the filters. For example, `[1,2,3]` would
            create a 2 bins: (1,2) and (2,3).
        order: bandpass filter order
    """

    def __init__(
        self, sample_rate: int, buffer_size: int, bin_edges: List[int], order: int
    ):
        self._buffer_size = buffer_size
        self._sample_rate = sample_rate

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

    # FIXME: make some types for all these
    def __call__(self, sample_block: np.ndarray) -> List[Dict[Tuple[int, int], float]]:
        """ get list of rms values for each bin

        An effort is made to reduce (eliminate?) transients present at the begining
            of the filter by prepending the previous input sample block to the one we
            care about.

        Args:
            sample_block: Input to be processed. MUST be the same len as `sample_rate`

        Returns:
            List[Dict[Tuple[int, int], float]]: list of bin-intensity mappings for
                each variable.
                Example: `[{(1,2):0.0, (2,3):1.9}, {(1,2):2.3, (2,3):4.6}]` is a list
                of 2 buffers with 2 bins.
        """

        assert len(sample_block) == self._sample_rate
        buffered_sample_block = self._buffer_sample_block(sample_block)

        shared_sample_block = np.ndarray(
            (len(buffered_sample_block),),
            dtype=np.float32,
            buffer=self.shared_sample_block_memory.buf,
        )

        shared_sample_block[:] = buffered_sample_block[:]

        # tell children to get to work
        for bandpass_filter in self._bandpass_filters.values():
            bandpass_filter()
        LOGGER.debug("all filters processing")

        # collect the results into a list
        bin_intensities_list: List[Tuple[Tuple[int, int], np.ndarray]] = []
        for bin_, bandpass_filter in self._bandpass_filters.items():
            intensity_list = []
            filtered_sample_block = bandpass_filter.result

            # split filtered samples into buffers
            for buffer in filtered_sample_block.reshape(
                int(len(filtered_sample_block) / self._buffer_size), self._buffer_size
            ):
                energy = float(dsp.get_rms(buffer))
                max_energy = bandpass_filter.max_energy_tracker(energy)
                if not max_energy:
                    intensity = 0
                else:
                    intensity = energy / max_energy
                intensity_list.append(intensity)
            bin_intensities_list.append((bin_, np.array(intensity_list)))
        LOGGER.debug("all results recieved")

        # filterbanks return a random order, sort them into a 2d array
        bin_intensities_list.sort()
        bin_intensities_array = np.zeros(
            (len(bin_intensities_list), len(bin_intensities_list[0][1])),
            dtype=bin_intensities_list[0][1].dtype,
        )
        for i, (_, intensities) in enumerate(bin_intensities_list):
            bin_intensities_array[i] = intensities[:]

        # create a bin-intensity mapping for each buffer
        bin_intensity_mapping_list = []
        for intensity_array in bin_intensities_array.swapaxes(0, 1):
            bin_intensity_mapping = {}
            for bin_, intensity in zip(self._bandpass_filters, intensity_array):
                bin_intensity_mapping[bin_] = float(intensity)
            bin_intensity_mapping_list.append(bin_intensity_mapping)

        return bin_intensity_mapping_list
