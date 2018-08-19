"""
tests for retriever.py
"""

from __future__ import absolute_import
import numpy as np
from src.retriever import AbstractRetriever, MPEGRetriever


class TestRetriever(AbstractRetriever):
    """Tests the Abstract Retriever"""

    @staticmethod
    def retrieve():
        """sends junk retrieval"""
        return np.array([[[]]])

    def reset(self):
        """resets"""
        pass


def test_abstract_retriever():
    """testing abstract retriever"""
    # pylint:disable=bare-except
    try:
        retriever = TestRetriever()
        retriever.reset()
        retriever.retrieve()
    except:
        assert False


def test_mpeg_retriever():
    """tests the mpeg retriever"""
    retriever = MPEGRetriever('./src/__tests__/test.mp4')
    assert retriever.retrieve().shape == (1444, 480, 270, 3)
