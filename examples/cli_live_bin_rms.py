""" View live (slightly delayed) graph of the RMS of each bin"""

import atexit
import logging
import multiprocessing as mp
import os
import sys
import time

import psutil  # type:ignore
import pymvf

logging.basicConfig(filename="pymvf.log", level=20)
LOGGER = logging.getLogger(__name__)


@atexit.register
def _killtree(including_parent: bool = True) -> None:
    """ Kill all children processes at script exit

    Mostly valuable in event of a crash, so PyAudio doesn't keep recording forever.

    Args:
        including_parent: if parent should also be killed when executed
    """
    LOGGER.critical("Stopping")
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()

    if including_parent:
        parent.kill()

    LOGGER.critical("Stopped")


def main() -> None:
    buffer_discard_qty = 10
    buffer_processor = pymvf.PyMVF(
        bin_edges=pymvf.dsp.generate_bin_edges(20, 20000, int(sys.argv[1])),
        buffer_discard_qty=10,
    )

    start_time = None
    while True:
        if start_time is None:
            buffer = buffer_processor()
            start_time = time.monotonic()

        real_time = time.monotonic() - start_time
        buffer = buffer_processor()

        LOGGER.info(f"plotting buffer {buffer.id}")

        # FIXME: something is wrong here, the delta should never be negative but is
        time_per_buffer = 512 / 44100
        plotted_buffers = buffer.id - buffer_discard_qty
        audio_time = (plotted_buffers * time_per_buffer) - time_per_buffer
        delta = real_time - audio_time

        # clear screen
        print(chr(27) + "[2J")
        # move cursor to (1,1)
        print(chr(27) + "[1;1f")
        print(
            f"Real Time: {round(real_time,2)}\n"
            f"Audio Time: {round(audio_time,2)}\n"
            f"Delta: {round(delta,2)}"
        )
        for bin_, energy in buffer.mono_bin_energy_mapping.items():
            print(f"{bin_}" + "-" * int(energy * 200))
        time.sleep(500 / 44100)

        # for bin_, bin_amplitude in buffer.mono_bin_energy_mapping.items():
        #     print(f"{bin_:}" + "-" * int(bin_amplitude * 100))


if __name__ == "__main__":
    main()
