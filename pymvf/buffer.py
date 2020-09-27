import audioop
from dataclasses import dataclass
from typing import List

import aubio
import numpy as np


class Buffer:

    raw_data: np.ndarray
    freqeuncies: np.ndarray
    amplitudes: np.ndarray
    rms: float

    def __init__(
        self,
        raw_data: np.ndarray,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        rms: float,
    ):
        self.raw_data = raw_data
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.rms = rms


class Processor:

    buffer_size: int
    sample_rate: int

    fft: aubio.fft
    filter_bank: aubio.filterbank

    def __init__(self, buffer_size: int, sample_rate: int):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate

        self.fft = aubio.fft(self.buffer_size)
        self.filter_bank = aubio.filterbank(48, self.buffer_size)
        self.filter_bank.set_power(2)

        self.freqeuncies = [
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

        self.filter_bank.set_triangle_bands(
            aubio.fvec(self.freqeuncies), self.sample_rate
        )

    def create_buffer(self, in_data) -> Buffer:
        mono = audioop.tomono(in_data, 4, 0.5, 0.5)
        data = np.frombuffer(mono, dtype=np.float32)

        fft = self.fft(data)
        amplitudes = np.around(self.filter_bank(fft), 2)
        frequencies = np.around(self.freqeuncies[1:-1], 2)
        rms = audioop.rms(in_data, 4)

        return Buffer(
            raw_data=data, frequencies=frequencies, amplitudes=amplitudes, rms=rms
        )

