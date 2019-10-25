"""
This file tests the MidiToNumpy class
"""
import unittest
import mido

from MidiNumpyConversion.MidiToNumpy import MidiToNumpy
from constants import NUM_TIMES


class TestMidiToNumpy(unittest.TestCase):
    """
    Tests the MidiToNumpy class
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.midipath = "midipath"
        self.numpypath = "numpypath"
        self.ppq = 960
        self.midi_time = self.ppq * 7  # The 7th quarter note
        self.tensor_time_4_4 = NUM_TIMES * 3 / 4
        self.tensor_time_3_4 = NUM_TIMES * 1 / 3
        self.tensor_time_2_4 = NUM_TIMES * 1 / 2
        self.mido_messages = [mido.MetaMessage("time_signature", time=0),
                              mido.MetaMessage("time_signature", time=100),
                              mido.MetaMessage("track_name", time=150),
                              mido.MetaMessage("time_signature", time=250)]
        self.abs_times = [0, 100, 250, 500]
        self.time_sig_times = [0, 100, 500]
        self.midifile = mido.MidiFile()
        self.midifile.tracks = [self.mido_messages]

    def setUp(self):
        """
        Creates a MidiToNumpy object to be used by other tests
        """
        self.mtn = MidiToNumpy(self.midipath, self.numpypath)

    def test_constructor(self):
        """
        Checks that the constructor assigns the midi_path
        and numpy_path correctly.
        """
        self.assertEqual(self.mtn.midi_path, self.midipath)
        self.assertEqual(self.mtn.numpy_path, self.numpypath)

    def test_time_conversion_4_4(self):
        """
        Checks that the midi_to_tensor_time function
        works correctly in 4/4 time.
        """
        tensor_time = self.mtn.midi_to_tensor_time(self.midi_time, self.ppq, 4)
        self.assertEqual(tensor_time, self.tensor_time_4_4)

    def test_time_conversion_3_4(self):
        """
        Checks that the midi_to_tensor_time function
        works correctly in 3/4 time.
        """
        tensor_time = self.mtn.midi_to_tensor_time(self.midi_time, self.ppq, 3)
        self.assertEqual(tensor_time, self.tensor_time_3_4)

    def test_time_conversion_2_4(self):
        """
        Checks that the midi_to_tensor_time function
        works correctly in 2/4 time.
        """
        tensor_time = self.mtn.midi_to_tensor_time(self.midi_time, self.ppq, 2)
        self.assertEqual(tensor_time, self.tensor_time_2_4)

    def test_relative_to_absolute_times(self):
        """
        Checks that the relative_to_absolute_times function
        works correctly for some mido MetaMessages.
        """
        abs_times = self.mtn.relative_to_absolute_times(self.mido_messages)
        abs_times = [m.time for m in abs_times]
        self.assertEqual(abs_times, self.abs_times)

    def test_get_time_signatures_message_types(self):
        """
        Checks that the get_time_signatures returns only
        time_signature type meta messages.
        """
        time_signatures = self.mtn.get_time_signatures(self.midifile)
        self.assertEqual(len(time_signatures), 3)
        for ts in time_signatures:
            self.assertEqual(ts.type, "time_signature")

    def test_get_time_signatures_message_times(self):
        """
        Checks that the get_time_signatures returns meta messagse in
        absolute times.
        """
        time_signatures = self.mtn.get_time_signatures(self.midifile)
        abs_times = [m.time for m in time_signatures]
        self.assertEqual(abs_times, self.time_sig_times)

    # TODO: integration tests of midi_file_to_numpy
    # TODO: functional tests of load_midis
