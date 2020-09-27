from dataclasses import dataclass
from typing import List

import aubio
import numpy as np

import audioop
from pymvf import (
    BUFFER_SIZE,
    SAMPLE_RATE,
)

FILTERBANK_BINS = [
    0,
    22,
    46,
    72,
    101,
    133,
    167,
    205,
    247,
    292,
    342,
    397,
    457,
    523,
    595,
    674,
    760,
    855,
    958,
    1072,
    1197,
    1333,
    1483,
    1647,
    1827,
    2023,
    2239,
    2475,
    2734,
    3018,
    3329,
    3670,
    4044,
    4453,
    4901,
    5393,
    5931,
    6521,
    7167,
    7876,
    8652,
    9503,
    10435,
    11456,
    12575,
    13802,
    15146,
    16618,
    18232,
    20000,
]

_fft: aubio.fft = aubio.fft(BUFFER_SIZE)
_filter_bank: aubio.filterbank = aubio.filterbank(48, BUFFER_SIZE)
_filter_bank.set_power(2)
_filter_bank.set_triangle_bands(aubio.fvec(FILTERBANK_BINS), SAMPLE_RATE)


class Buffer:
    def __init__(self, timestamp: float, latency: float, stereo_buffer: bytes):
        self.timestamp = timestamp
        self.latency = latency

        self.stereo_buffer: bytes = stereo_buffer
        self.mono_buffer: bytes = audioop.tomono(self.stereo_buffer, 4, 0.5, 0.5)

        self.mono_array = np.frombuffer(self.mono_buffer, dtype=np.float32)
        self.stereo_array = np.frombuffer(self.stereo_buffer, dtype=np.float32)

        self.left_channel_array = self.stereo_array[0::2]
        self.right_channel_array = self.stereo_array[1::2]

        self.left_channel_filterbank = self._get_filterbank(self.left_channel_array)
        self.right_channel_filterbank = self._get_filterbank(self.right_channel_array)

        self.left_rms: float = audioop.rms(self.left_channel_array.tobytes(), 4)
        self.right_rms: float = audioop.rms(self.right_channel_array.tobytes(), 4)

    def _get_filterbank(self, input_array: np.ndarray) -> tuple:
        fft = _fft(input_array)
        # we don't need much precision at all
        amplitudes = np.around(_filter_bank(fft))
        # don't include first and last bins

        return np.column_stack((FILTERBANK_BINS, amplitudes))
