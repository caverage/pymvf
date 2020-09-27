import signal
from typing import List

import numpy as np
import pyaudio

from pymvf.buffer import Buffer, Processor


class PyMVF:

    buffer_size: int
    sample_rate: int
    channels: int
    processor: Processor
    buffers: List[Buffer]

    def __init__(self, buffer_size: int, sample_rate: int, channels: int):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.processor = Processor(
            buffer_size=self.buffer_size, sample_rate=self.sample_rate
        )
        self.buffers = []

    def pyaudio_callback(self, in_data, frame_count, time_info, flag):
        buffer = self.processor.create_buffer(in_data)

        self.buffers.append(buffer)
        if len(self.buffers) >= 100:
            self.buffers.pop()

        print(chr(27) + "[2J")
        print("\033[H")

        maxpos = np.argmax(buffer.amplitudes)
        print(f"Active Frequency: {buffer.frequencies[maxpos]}")
        print(f"RMS: {buffer.rms}")

        return (None, pyaudio.paContinue)

    def run(self):
        pa = pyaudio.PyAudio()
        pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self.pyaudio_callback,
        )

        signal.pause()


if __name__ == "__main__":
    pymvf = PyMVF(buffer_size=2048, sample_rate=44100, channels=2)
    pymvf.run()
