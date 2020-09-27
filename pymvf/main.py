import audioop
import signal

import aubio
import numpy as np
import pyaudio

SAMPLE_RATE = 48000

BUFFER_SIZE = 2048
BEAT_BUFFER_SIZE = 128

tempo = aubio.tempo("default", BUFFER_SIZE * 2, BUFFER_SIZE, SAMPLE_RATE)


def process_amps(in_data):
    mono = audioop.tomono(in_data, 4, 0.5, 0.5)
    data = np.frombuffer(mono, dtype=np.float32)
    fft = aubio.fft(BUFFER_SIZE)(data)
    fb = aubio.filterbank(400, BUFFER_SIZE)
    fb.set_power(2)
    freqs = np.linspace(0, 20_000, 402)
    fb.set_triangle_bands(aubio.fvec(freqs), SAMPLE_RATE)

    output = np.around(fb(fft), 2)
    freqs = np.around(freqs[1:-1], 2)

    maxpos = np.argmax(output)
    print(f"Active Frequency: {freqs[maxpos]}")

    # for bin, amplitude in np.column_stack((freqs, output)):
    #    if amplitude > 1:
    #        print(f"{bin} : {amplitude}")


def process_channels(in_data):
    data = np.frombuffer(in_data, dtype=np.float32)
    left = data[0::2]
    right = data[1::2]

    print(f"Left: {np.array_str(left)}")
    print(f"Right: {np.array_str(right)}")


def process_beat_detection(in_data):
    mono = audioop.tomono(in_data, 4, 0.5, 0.5)
    data = np.frombuffer(mono, dtype=np.float32)
    beat = tempo(data)
    print(f"BPM: {tempo.get_bpm()}")


def callback(in_data, frame_count, time_info, flag):
    print(chr(27) + "[2J")
    print("\033[H")
    process_channels(in_data)
    process_amps(in_data)
    process_beat_detection(in_data)
    return (None, pyaudio.paContinue)


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
