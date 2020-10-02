""" Plot energy of each bin

Args:
    1 (float): time in seconds to wait between plots. values > 0.3 work best
"""

import multiprocessing as mp
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import pymvf

OUTPUT_QUEUE: mp.Queue = mp.Queue()

PYMVF_PROCESS = mp.Process(target=pymvf.PyMVF, args=(OUTPUT_QUEUE,))
PYMVF_PROCESS.start()


def main() -> None:
    plt.rcdefaults()

    # wait for pyaudio to actually start sending us some usable data
    buffer_id = 0
    while buffer_id < 5:
        buffer_id = OUTPUT_QUEUE.get().id

    print(f"Starting")
    # init all the variables with the first buffer
    buffer = OUTPUT_QUEUE.get()
    max_left = buffer.left_channel_filterbank
    max_right = buffer.right_channel_filterbank

    next_plot = time.monotonic() + 1
    plt.ion()
    # create plot
    fig, axis = plt.subplots()

    # disable exponents

    index = np.arange(len(pymvf.FILTERBANK_BINS[1:-1]))
    bar_width = 0.35
    opacity = 0.8

    plt.xlabel("Frequency Bin")
    plt.ylabel("Energy")
    plt.title(f"Energy Per Bin Every {float(sys.argv[1])} Seconds")
    plt.xticks(index + bar_width, pymvf.FILTERBANK_BINS[1:-1], rotation=45)

    left_channel = plt.bar(
        index,
        buffer.left_channel_filterbank,
        bar_width,
        alpha=opacity,
        color="b",
        label="Left Channel",
    )

    right_channel = plt.bar(
        index + bar_width,
        buffer.right_channel_filterbank,
        bar_width,
        alpha=opacity,
        color="g",
        label="Right Channel",
    )

    plt.legend()
    plt.tight_layout()

    previous_buffer_id = buffer.id

    max_energy = 0
    while True:
        buffer = OUTPUT_QUEUE.get()

        if next_plot < time.monotonic():
            next_plot = time.monotonic() + float(sys.argv[1])
            print(
                f"plotting: {buffer.id}\nskipped: {buffer.id - previous_buffer_id - 1}"
            )
            previous_buffer_id = buffer.id
            next_plot = time.monotonic() + float(sys.argv[1])

            for rect, energy in zip(left_channel, buffer.left_channel_filterbank):
                rect.set_height(energy)
            for rect, energy in zip(right_channel, buffer.right_channel_filterbank):
                rect.set_height(energy)

            # get max energy value to set scale
            if (
                np.amax(
                    np.concatenate(
                        (
                            buffer.left_channel_filterbank,
                            buffer.right_channel_filterbank,
                        )
                    )
                )
                > max_energy
            ):
                max_energy = np.amax(
                    np.concatenate(
                        (
                            buffer.left_channel_filterbank,
                            buffer.right_channel_filterbank,
                        )
                    )
                )

            axis.set_ylim([0, max_energy * 1.1])
            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == "__main__":
    main()
