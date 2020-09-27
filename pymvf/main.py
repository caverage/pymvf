import signal
import time

import numpy as np
import pyaudio

SAMPLE_RATE = 44100
TIME_STEP = 1 / SAMPLE_RATE

BUFFER_SIZE = 2048

NEXT_FFT_TIME = time.monotonic() + 1


def callback(in_data, frame_count, time_info, flag):
    print(chr(27) + "[2J")
    print("\033[H")
    data = np.fromstring(in_data, dtype=np.float32)
    ch1 = data[0::2]
    ch2 = data[1::2]
    print(ch1)
    print(ch2)
    # return (in_data, recording)

    return None, pyaudio.paContinue


def main():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE,
        stream_callback=callback,
    )

    signal.pause()


if __name__ == "__main__":
    main()
