"""
This file tests the NumpyToMidi class
"""
import unittest
import mido

from MidiNumpyConversion.NumpyToMidi import NumpyToMidi
from constants import NUM_TIMES


class TestNumpyToMidi(unittest.TestCase):
    """
    Tests the NumpyToMidi class
    """
    def __init__(self, *args):
        super().__init__(*args)
        ...

    def setUp(self):
        """
        TODO: implement
        """
