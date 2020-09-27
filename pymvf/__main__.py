import multiprocessing as mp
import signal
import time
from typing import Optional

import pyaudio

from pymvf import (
    BUFFER_SIZE,
    SAMPLE_RATE,
)

BUFFER_QUEUE: Optional[mp.Queue] = None
STREAM: Optional[pyaudio.Stream] = None

time.perf_counter()  # initialize the perf_counter


def _callback(in_data, frame_count, time_info, flag):
    input_latency = STREAM.get_input_latency()
    timestamp = time.perf_counter() - input_latency

    BUFFER_QUEUE.put((timestamp, input_latency, in_data))

    return (None, pyaudio.paContinue)


def input_stream():
    global STREAM
    STREAM = pyaudio.PyAudio().open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE,
        stream_callback=_callback,
    )

    signal.pause()


def main():
    global BUFFER_QUEUE
    BUFFER_QUEUE = mp.Queue()

    input_stream_process = mp.Process(target=input_stream)
    input_stream_process.start()

    time.sleep(5)
    while True:
        timestamp, input_latency, buffer = BUFFER_QUEUE.get()
        print(timestamp)


if __name__ == "__main__":
    main()
