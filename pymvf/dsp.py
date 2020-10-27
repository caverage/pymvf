""" Module for signal processing helpers and utilities"""

from typing import List, Tuple

import numpy as np  # type:ignore


def erb_from_freq(freq: int) -> float:
    """ Get equivalent rectangular bandwidth from the given frequency.

    See: https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth

    Args:
        freq: input frequency

    Returns:
        int: cam
    """

    return float(9.265 * np.log(1 + np.divide(freq, 24.7 * 9.16)))


def generate_bin_edges(low_freq: int, high_freq: int, count: int) -> List[int]:
    """ Bin sizes designed around Equivalent Rectangular Bandwidth

    NOTE: this becomes less accurate as the bin values increase,
        but should be good enough (TM)

    Args:
        low_freq: where to start the bins (> 0)
        high_freq: where to end the bins (<= 20,000)
        count: number of bin edges to generate

    Returns:
        List[int]: bin seperators in Hz
    """

    bin_edges = []
    cams = np.linspace(erb_from_freq(low_freq), erb_from_freq(high_freq), count)
    for i, cam in enumerate(cams):
        if not i:
            # this is probably not nessesary, but better safe than sorry?
            bin_edges.append(low_freq)
        elif i == len(cams) - 1:
            bin_edges.append(high_freq)
        else:
            bin_edges.append(round(10 ** (cam / 21.4) / 0.00437 - 1 / 0.00437))

    return bin_edges


def get_rms(sample_block: np.array) -> float:
    """ Get the RMS of an array of audio sample_block

    Args:
        sample_block: the sample_block to get the RMS from

    Returns:
        float: the RMS
    """
    return np.sqrt(np.mean(sample_block ** 2))  # type:ignore


def split_channels(stereo_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ split the channels of a stereo array

    Args:
        stereo_array: the stereo array to split

    Returns:
        Tuple[np.ndarray, np.ndarray]: left channel, right channel
    """

    stereo_2d = np.reshape(stereo_array, (int(len(stereo_array) / 2), 2))

    left_channel_array = stereo_2d[:, 0].copy()
    right_channel_array = stereo_2d[:, 1].copy()

    return left_channel_array, right_channel_array
