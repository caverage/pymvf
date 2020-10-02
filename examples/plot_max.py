""" Plot max energy of each bin within given timeframe.

Args:
    1: time in seconds to record before plotting
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

    print("Starting")
    # init all the variables with the first buffer
    buffer = OUTPUT_QUEUE.get()
    max_left = buffer.left_channel_filterbank
    max_right = buffer.right_channel_filterbank

    end_time = time.monotonic() + float(sys.argv[1])
    while end_time > time.monotonic():
        buffer = OUTPUT_QUEUE.get()

        max_left = np.maximum(max_left, buffer.left_channel_filterbank)
        max_right = np.maximum(max_right, buffer.right_channel_filterbank)

    print(f"Got {buffer.id - 5} buffers, plotting.")
    # create plot
    fig, axis = plt.subplots()

    # disable exponents
    axis.ticklabel_format(style="plain")

    index = np.arange(len(pymvf.FILTERBANK_BINS[1:-1]))
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(
        index, max_left, bar_width, alpha=opacity, color="b", label="Max Left Channel"
    )

    rects2 = plt.bar(
        index + bar_width,
        max_right,
        bar_width,
        alpha=opacity,
        color="g",
        label="Max Right Channel",
    )

    plt.xlabel("Frequency Bin")
    plt.ylabel("Energy")
    plt.title(f"Max Energy Per Bin Over {sys.argv[1]} Seconds")
    plt.xticks(index + bar_width, pymvf.FILTERBANK_BINS[1:-1], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
