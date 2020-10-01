""" Create CSV from output buffer of both channels individually.

One bin per column, per channel.

Args:
    1: output file
"""

import csv
import multiprocessing as mp
import sys
import time
from typing import List

import numpy as np

import pymvf

OUTPUT_QUEUE: mp.Queue = mp.Queue()

PYMVF_PROCESS = mp.Process(target=pymvf.PyMVF, args=(OUTPUT_QUEUE,))
PYMVF_PROCESS.start()

next_print = time.monotonic() + 1


def main() -> None:
    with open(sys.argv[1], "w", newline="") as output_csv:
        fieldnames: List[str] = [
            "timestamp",
            "latency",
            "channel",
            "rms",
        ] + pymvf.FILTERBANK_BINS[1:-1]
        csv_writer = csv.DictWriter(output_csv, fieldnames=fieldnames, delimiter=",")
        csv_writer.writeheader()

        while True:
            buffer = OUTPUT_QUEUE.get()

            left_channel = {
                bin: energy for [bin, energy] in buffer.left_channel_filterbank
            }
            csv_writer.writerow(
                {
                    "timestamp": buffer.timestamp,
                    "latency": buffer.latency,
                    "channel": "left",
                    "rms": buffer.right_rms,
                    **left_channel,
                }
            )

            right_channel = {
                bin: energy for [bin, energy] in buffer.right_channel_filterbank
            }
            csv_writer.writerow(
                {
                    "timestamp": buffer.timestamp,
                    "latency": buffer.latency,
                    "channel": "right",
                    "rms": buffer.right_rms,
                    **right_channel,
                }
            )


if __name__ == "__main__":
    main()
