from dataclasses import dataclass
from typing import Callable, Tuple

import aubio
import numpy as np

from pymvf import FILTERBANK_BINS


class FilterBank:
    def __init__(self, sample_rate: int, buffer_size: int) -> None:
        self._fft = aubio.fft(buffer_size)
        self._filterbank = aubio.filterbank(len(FILTERBANK_BINS) - 2, buffer_size)
        self._filterbank.set_power(3)
        self._filterbank.set_triangle_bands(aubio.fvec(FILTERBANK_BINS), sample_rate)

        coefficients = self._filterbank.get_coeffs()

        # increase the relative power of the higher bins
        for i, coefficient in enumerate(coefficients):
            # first 2 coefficients stay the same
            coefficients[i] = coefficient * (i ** 2.2)

        self._filterbank.set_coeffs(coefficients)

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        fft = self._fft(input_array)
        # we don't need much precision at all
        energy = self._filterbank(fft).astype(np.uint32)

        return energy


class Buffer:
    def __init__(
        self,
        buffer_id: int,
        timestamp: float,
        latency: float,
        stereo_buffer: bytes,
        filterbank: FilterBank,
    ):
        self.id = buffer_id
        self.timestamp = timestamp
        self.latency = latency

        self.stereo_buffer: bytes = stereo_buffer

        self.stereo_array = np.frombuffer(self.stereo_buffer, dtype=np.float32)

        stereo_2d = np.reshape(self.stereo_array, (int(len(self.stereo_array) / 2), 2))

        self.left_channel_array = stereo_2d[:, 0].copy()
        self.right_channel_array = stereo_2d[:, 1].copy()

        self.mono_array = np.add(
            self.left_channel_array / 2, self.right_channel_array / 2
        )

        self.left_channel_filterbank = filterbank(self.left_channel_array)
        self.right_channel_filterbank = filterbank(self.right_channel_array)

        # https://stackoverflow.com/a/9763652/1342874
        self.left_rms: float = np.sqrt(
            sum(self.left_channel_array * self.left_channel_array)
            / len(self.left_channel_array)
        )
        self.right_rms: float = np.sqrt(
            sum(self.right_channel_array * self.right_channel_array)
            / len(self.right_channel_array)
        )
