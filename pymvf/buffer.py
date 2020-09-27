import audioop
from dataclasses import dataclass
from typing import List

import aubio
import numpy as np

from pymvf import BUFFER_SIZE, SAMPLE_RATE

_bins = [
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
_filter_bank.set_triangle_bands(aubio.fvec(_bins), SAMPLE_RATE)


class Buffer:
    def __init__(self, raw_data: bytes):
        self.raw_data: bytes = raw_data

        bins, amplitudes = self.calculate_bins(self.raw_data)
        self.bins: np.ndarray = bins
        self.amplitudes: np.ndarray = amplitudes

        self.rms: float = audioop.rms(self.raw_data, 4)

    def calculate_bins(self, in_data: bytes) -> tuple:
        mono = audioop.tomono(in_data, 4, 0.5, 0.5)
        data = np.frombuffer(mono, dtype=np.float32)

        fft = _fft(data)
        amplitudes = np.around(_filter_bank(fft), 2)
        bins = np.around(_bins[1:-1], 2)

        return (bins, amplitudes)
