import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import aubio  # type:ignore
import numpy as np  # type:ignore

from . import dsp

LOGGER = logging.getLogger(__name__)


@dataclass
class Beat:
    """ Beat information

    Attributes:
        confidence: the confidence of the beat
        bpm: the current beat per minute
    """

    confidence: float
    bpm: float


@dataclass
class BufferStub:
    """ Stub for holding data for a single buffer before the bin RMS is done.

    Attributes:
        id: identification number of the buffer
        timestamp: time that the buffer was recieved
        mono_rms: root mean square (continuous power) of buffer
        left_rms: root mean square (continuous power) of left channel
        right_rms: root mean square (continuous power) of right channel
        beat: beat information, or none if no beat for this buffer
    """

    id: int
    timestamp: float

    mono_rms: float
    left_rms: float
    right_rms: float

    beat: Optional[Beat]


@dataclass
class Buffer(BufferStub):
    """ Class for holding data for a single buffer.

    See: BufferStub.

    Attributes:
        mono_intensity: average intensity of channel
        left_intensity: average intensity of channel
        right_intensity: average intensity of channel
        mono_bin_intensity_mapping: mono bin intensity
        left_bin_intensity_mapping: bin intensity of left channel
        right_bin_intensity_mapping: bin intensity of right channel
    """

    mono_intensity: Dict[Tuple[int, int], np.ndarray]
    left_intensity: Dict[Tuple[int, int], np.ndarray]
    right_intensity: Dict[Tuple[int, int], np.ndarray]

    mono_bin_intensity_mapping: Dict[Tuple[int, int], np.ndarray]
    left_bin_intensity_mapping: Dict[Tuple[int, int], np.ndarray]
    right_bin_intensity_mapping: Dict[Tuple[int, int], np.ndarray]


def create_buffer_stub(
    buffer_id: int,
    timestamp: float,
    mono_sample_buffer: np.ndarray,
    left_sample_buffer: np.ndarray,
    right_sample_buffer: np.ndarray,
    beat_detector: aubio.tempo,
) -> BufferStub:
    """ Create a Buffer

    Args:
        buffer_id: the id of the input buffer
        timestamp: when the buffer was returned by portaudio
        stereo_buffer: the two channel buffer
        left_sample_buffer: left channel sample buffer
        right_sample_buffer: right channel sample buffer
        mono_sample_buffer: mono sample buffer

    Returns:
        Buffer: the buffer object
    """

    mono_rms = dsp.get_rms(left_sample_buffer)
    left_rms = dsp.get_rms(right_sample_buffer)
    right_rms = dsp.get_rms(mono_sample_buffer)

    beat_dict = {}
    beat_detector_result = beat_detector(mono_sample_buffer)
    if beat_detector_result[0] > 0:
        beat_dict["confidence"] = beat_detector.get_confidence()
        beat_dict["bpm"] = beat_detector.get_bpm()
        beat: Optional[Beat] = Beat(**beat_dict)
    else:
        beat = None

    return BufferStub(
        id=buffer_id,
        timestamp=timestamp,
        mono_rms=mono_rms,
        left_rms=left_rms,
        right_rms=right_rms,
        beat=beat,
    )
