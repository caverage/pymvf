import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple

import aubio  # type:ignore
import numpy as np  # type:ignore

from pymvf import signal_processing

LOGGER = logging.getLogger(__name__)


@dataclass
class Buffer:
    """ Class for holding data for a single buffer

    Attributes:
        id: identification number of the buffer
        timestamp: time that the buffer was recieved
        filterbank: mono filterbank
        left_filterbank: filterbank of left channel
        right_filterbank: filterbank of right channel
        rms: root mean square (continuous power) of buffer
        left_rms: root mean square (continuous power) of left channel
        right_rms: root mean square (continuous power) of right channel
        beat: if the Buffer is a beat
    """

    id: int
    timestamp: float

    mono_bin_rms: Dict[Tuple[int, int], np.ndarray]
    left_bin_rms: Dict[Tuple[int, int], np.ndarray]
    right_bin_rms: Dict[Tuple[int, int], np.ndarray]

    mono_rms: float
    left_rms: float
    right_rms: float

    beat: bool


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


def create_buffer(
    buffer_id: int,
    timestamp: float,
    stereo_buffer: bytes,
    left_calculate_bin_rms: signal_processing.CalculateBinRMS,
    right_calculate_bin_rms: signal_processing.CalculateBinRMS,
) -> Buffer:
    """ Create a Buffer

    Args:
        buffer_id: the id of the input buffer
        timestamp: when the buffer was returned by portaudio
        latency: the latency of the audio system as reported by portaudio
        stereo_buffer: the two channel buffer

    Returns:
        Buffer: the buffer object
    """

    stereo_array = np.frombuffer(stereo_buffer, dtype=np.float32)

    left_channel_array, right_channel_array = split_channels(stereo_array)
    mono_array = np.add(left_channel_array / 2, right_channel_array / 2)

    left_bin_rms = left_calculate_bin_rms(left_channel_array)
    right_bin_rms = right_calculate_bin_rms(right_channel_array)
    mono_bin_rms = {}
    for bin_ in left_bin_rms.keys():
        mono_bin_rms[bin_] = np.add(left_bin_rms[bin_], right_bin_rms[bin_]) / 2

    mono_rms = signal_processing.get_rms(mono_array)
    left_rms = signal_processing.get_rms(left_channel_array)
    right_rms = signal_processing.get_rms(right_channel_array)

    beat = None

    return Buffer(
        id=buffer_id,
        timestamp=timestamp,
        mono_bin_rms=mono_bin_rms,
        left_bin_rms=left_bin_rms,
        right_bin_rms=right_bin_rms,
        mono_rms=mono_rms,
        left_rms=left_rms,
        right_rms=right_rms,
        beat=beat,
    )


# class Buffer:
#     def __init__(
#         self,
#         buffer_id: int,
#         timestamp: float,
#         latency: float,
#         stereo_buffer: bytes,
#         filterbank: signal.FilterBank,
#         sample_rate: int,
#         buffer_size: int,
#     ):
#         self.id = buffer_id
#         self.timestamp = timestamp
#         self.latency = latency
#
#         self.stereo_buffer: bytes = stereo_buffer
#
#         self.stereo_array = np.frombuffer(self.stereo_buffer, dtype=np.float32)
#
#         stereo_2d = np.reshape(self.stereo_array, (int(len(self.stereo_array) / 2), 2))
#
#         self.left_channel_array = stereo_2d[:, 0].copy()
#         self.right_channel_array = stereo_2d[:, 1].copy()
#
#         self.mono_array = np.add(
#             self.left_channel_array / 2, self.right_channel_array / 2
#         )
#
#         self.left_channel_filterbank = filterbank(self.left_channel_array)
#         self.right_channel_filterbank = filterbank(self.right_channel_array)
#
#         # https://stackoverflow.com/a/9763652/1342874
#         self.left_rms: float = np.sqrt(
#             sum(self.left_channel_array * self.left_channel_array)
#             / len(self.left_channel_array)
#         )
#         self.right_rms: float = np.sqrt(
#             sum(self.right_channel_array * self.right_channel_array)
#             / len(self.right_channel_array)
#         )
