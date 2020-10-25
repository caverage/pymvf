""" test the _BufferSampleBlock class"""
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest

import pymvf


def test_simple_no_remainder():
    buffer_sample_block = pymvf.signal_processing._BufferSampleBlock(10, 1)

    for _ in range(100):
        assert len(buffer_sample_block(np.zeros(10))) == 19


def test_simple_remainder_divisible():
    """ block sans-transient-prevention-buffer is divisible by the buffer size"""
    buffer_sample_block = pymvf.signal_processing._BufferSampleBlock(10, 3)

    for _ in range(100):
        assert len(buffer_sample_block(np.zeros(10))[10 - 3 :]) % 3 == 0


TEST_DATA = [
    (51, 3, None, does_not_raise()),
    (420, 13, None, does_not_raise()),
    (44100, 512, None, does_not_raise()),
]


@pytest.mark.parametrize("sample_rate,buffer_size,expected,raises", TEST_DATA)
def test_all_values_are_accounted_for(sample_rate, buffer_size, expected, raises):
    with raises:
        buffer_sample_block = pymvf.signal_processing._BufferSampleBlock(
            sample_rate, buffer_size
        )

        pseudo_blocks = np.linspace(
            1, sample_rate * buffer_size * 10, sample_rate * buffer_size * 10
        ).reshape((int((sample_rate * buffer_size * 10) / sample_rate), sample_rate))

        total_processed = 0
        for block in pseudo_blocks:
            buffered_block = buffer_sample_block(block)
            assert not len(buffered_block) > len(block) * 2

            block_to_process = buffered_block[sample_rate - buffer_size :]
            total_processed += len(block_to_process)

            assert not len(block_to_process) % buffer_size

            # more or less check that all numbers are consecutive, not just sequential
            # ensure that nothing is out of order
            assert np.array_equal(block_to_process, np.sort(block_to_process))
            assert (
                block_to_process[-1:]
                == block_to_process[:1] + len(block_to_process) - 1
            )

        assert total_processed == len(pseudo_blocks.reshape(-1))
