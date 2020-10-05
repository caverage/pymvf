import multiprocessing as mp
import signal
import time
from ctypes import c_longlong
from typing import Optional, Tuple

import numpy as np  # type: ignore
import pyaudio  # type: ignore

# Bin sizes designed around Equivalent Rectangular Bandwidth
# Calculated with:
# >>> import numpy as np
# >>> bins = []
# >>> for cam in np.linspace(0, 41.65407847127975, 50):
# ...     bins.append(round(10**(cam/21.4)/0.00437 - 1/0.00437))
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
            # pyaudio does not give good date for the first several buffers
            # a conservative amount of delay is introduced here
            if buffer_id < 10 or (buffer_id * self.buffer_size) < 5000:
                continue

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
