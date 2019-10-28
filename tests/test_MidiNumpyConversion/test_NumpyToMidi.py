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
        self.ntm = NumpyToMidi()

    def test_add_message_on(self):
        """
        Checks that the add_message function adds the note_on
        message correctly.
        """
        ...

    def test_add_message_off(self):
        """
        Checks that the add_message function adds the note_off
        message correctly.
        """
        ...

    def test_measure_to_messages_adds_note_on(self):
        """
        Checks that the measure_to_messages function adds the note_on
        message correctly.
        """
        ...

    def test_measure_to_messages_adds_note_off(self):
        """
        Checks that the measure_to_messages function adds the note_off
        message correctly.
        """
        ...

    def test_measure_to_messages_counts_time(self):
        """
        Checks that the measure_to_messages function returns the time
        between the last note in the measure and the end of it correctly.
        """
        ...

    def test_measure_to_messages_counts_active_notes(self):
        """
        Checks that the measure_to_messages function returns the number
        of active notes correctly.
        """
        ...

    def test_add_meta_messages_prepends_messages(self):
        """
        Checks that the add_meta_messages function prepends the
        correct meta messages to a midi track correctly.
        """
        ...

    def test_add_conductor_track(self):
        """
        Checks that the add_conductor_track function adds a track
        to the beginning of a midofile's tracks that contains a 4/4
        time signature message.
        """
        ...
