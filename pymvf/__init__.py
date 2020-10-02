import multiprocessing as mp
import signal
import time
from ctypes import c_longlong
from typing import Optional, Tuple

import pyaudio  # type: ignore

# Bin sizes designed around Equivalent Rectangular Bandwidth
# Calculated with:
# >>> import numpy as np
# >>> bins = []
# >>> for cam in np.linspace(0, 41.65407847127975, 49):
# ...     bins.append(round(10**(cam/21.4)/0.00437 - 1/0.00437))
FILTERBANK_BINS = [
    0,
    22,
    47,
    74,
    104,
    136,
    172,
    211,
    254,
    301,
    353,
    410,
    473,
    542,
    617,
    700,
    791,
    890,
    1000,
    1120,
    1252,
    1397,
    1556,
    1731,
    1923,
    2133,
    2364,
    2618,
    2897,
    3203,
    3539,
    3907,
    4312,
    4757,
    5245,
    5780,
    6368,
    7014,
    7723,
    8501,
    9356,
    10294,
    11323,
    12454,
    13695,
    15058,
    16554,
    18197,
    20000,
]

# FILTERBANK_BINS needs defined before these imports
from . import buffer  # isort:skip


class PyMVF:
    def __init__(self, output_queue: mp.Queue):
        self.sample_rate = 44100
        self.buffer_size = 512

        self._buffer_queue: mp.Queue = mp.Queue()

        self._manager = mp.Manager()
        self._lock = self._manager.Lock()

        # 'Q' | unsigned long long
        self.buffer_counter = self._manager.Value("Q", 0)

        self._input_stream_process = mp.Process(target=self._input_stream)
        self._input_stream_process.start()

        self._filterbank = buffer.FilterBank(self.sample_rate, self.buffer_size)

        while True:
            buffer_id, timestamp, latency, stereo_buffer = self._buffer_queue.get()
            output_queue.put(
                buffer.Buffer(
                    buffer_id, timestamp, latency, stereo_buffer, self._filterbank
                )
            )

    def _input_stream(self) -> None:
        self.stream = pyaudio.PyAudio().open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self._callback,
        )
        signal.pause()

    def _callback(self, in_data, frame_count, time_info, flag):  # pylint: disable=W0613
        with self._lock:
            self.buffer_counter.value += 1
        timestamp = time.perf_counter()
        input_latency = self.stream.get_input_latency()

        self._buffer_queue.put(
            (self.buffer_counter.value, timestamp, input_latency, in_data)
        )

        return (None, pyaudio.paContinue)
