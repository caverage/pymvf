""" View live (slightly delayed) graph of the RMS of each bin"""

import atexit
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import numpy as np  # type:ignore
import psutil  # type:ignore

import pymvf

logging.basicConfig(filename="cli_live_bin_rms.log", level=20)
LOGGER = logging.getLogger(__name__)


@atexit.register
def _killtree(including_parent: bool = True) -> None:
    """ Kill all children processes at script exit

    Mostly valuable in event of a crash, so PyAudio doesn't keep recording forever.

    Args:
        including_parent: if parent should also be killed when executed
    """

    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()

    if including_parent:
        parent.kill()


def main() -> None:
    buffer_discard_qty = 10
    output_queue: mp.Queue = mp.Queue()

    # we drop a buffer later, so compensate for that in the buffer_discard_qty
    pymvf_process = pymvf.Process(
        target=pymvf.PyMVF,
        args=(
            pymvf.signal_processing.generate_bin_edges(20, 20000, int(sys.argv[1])),
            output_queue,
            buffer_discard_qty - 1,
        ),
    )
    pymvf_process.start()

    rolling_delays = [
        deque([], maxlen=5),
        deque([], maxlen=50),
        deque([], maxlen=500),
    ]
    time_per_buffer = 512 / 44100
    start_time = None
    while True:
        if start_time is None:
            _ = output_queue.get()
            start_time = time.monotonic()
            previous_time = start_time

        buffer = output_queue.get()
        current_time = time.monotonic()
        delay = current_time - previous_time
        previous_time = current_time

        for rolling_delay in rolling_delays:
            rolling_delay.append(delay)

        plotted_buffers = buffer.id - buffer_discard_qty
        audio_time = (plotted_buffers * time_per_buffer) - time_per_buffer
        delta = (current_time - start_time) - audio_time

        # clear screen
        print(chr(27) + "[2J")
        # move cursor to (1,1)
        print(chr(27) + "[1;1f")

        print(
            f"Buffer: {buffer.id - buffer_discard_qty}\n"
            f"Real Time: {round(current_time-start_time,3)}\n"
            f"Audio Time: {round(audio_time,2)}\n"
            f"Delta: {round(delta,2)}\n"
            f"Break Even:{round(time_per_buffer,5)}\n"
            f"Delay 1: {round(delay,5)}\n"
            f"Delay 1: {round(delay,5)}\n"
            f"Delay 5: {round(np.sum(rolling_delays[0])/len(rolling_delays[0]),5)}\n"
            f"Delay 50: {round(np.sum(rolling_delays[1])/len(rolling_delays[1]),5)}\n"
            f"Delay 500: {round(np.sum(rolling_delays[2])/len(rolling_delays[2]),5)}\n"
        )


if __name__ == "__main__":
    main()
