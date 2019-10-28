"""
This file tests the NumpyToMidi class
"""
import unittest
import mido
import numpy as np

from MidiNumpyConversion.NumpyToMidi import NumpyToMidi
from constants import NUM_TIMES, NUM_NOTES, PPQ


class TestNumpyToMidi(unittest.TestCase):
    """
    Tests the NumpyToMidi class
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.note_on_msg = "note_on"
        self.note_off_msg = "note_off"
        self.track = mido.MidiTrack()
        self.track.append(mido.Message(self.note_on_msg, note=0))
        self.note_pitch = 69
        self.midi_note_time = PPQ * 3.5
        self.np_note_time = int(NUM_TIMES * 7/8)
        self.first_midi_note_time = 0
        self.first_np_note_time = 0
        self.measure = np.zeros((NUM_TIMES, NUM_NOTES))
        self.measure[self.np_note_time, self.note_pitch] = 1
        self.first_measure = np.zeros((NUM_TIMES, NUM_NOTES))
        self.first_measure[self.first_np_note_time, self.note_pitch] = 1
        self.time_to_measure_end = 4 * PPQ - self.midi_note_time
        self.midifile = mido.MidiFile()
        self.midifile.tracks = self.track

    def setUp(self):
        """
        Creates a NumpyToMidi object to be used by other tests.
        """
        self.ntm = NumpyToMidi()

    def test_add_message_on(self):
        """
        Checks that the add_message function adds the note_on
        message correctly.
        """
        track = self.ntm.add_message(self.track, self.note_pitch,
                                     self.midi_note_time, msg=self.note_on_msg)
        self.assertEqual(track[-1].note, self.note_pitch)
        self.assertEqual(track[-1].time, self.midi_note_time)
        self.assertEqual(track[-1].type, self.note_on_msg)

    def test_add_message_off(self):
        """
        Checks that the add_message function adds the note_off
        message correctly.
        """
        track = self.ntm.add_message(self.track, self.note_pitch,
                                     self.midi_note_time,
                                     msg=self.note_off_msg)
        self.assertEqual(track[-1].note, self.note_pitch)
        self.assertEqual(track[-1].time, self.midi_note_time)
        self.assertEqual(track[-1].type, self.note_off_msg)

    def test_measure_to_messages_adds_note_on(self):
        """
        Checks that the measure_to_messages function adds the note_on
        message correctly.
        """
        track, _, _ = self.ntm.measure_to_messages(self.measure, 0, [])
        self.assertEqual(track[-1].note, self.note_pitch)
        self.assertEqual(track[-1].time, self.midi_note_time)
        self.assertEqual(track[-1].type, self.note_on_msg)

    def test_measure_to_messages_adds_note_off(self):
        """
        Checks that the measure_to_messages function adds the note_off
        message correctly.
        """
        track, _, _ = self.ntm.measure_to_messages(self.first_measure, 0, [])
        self.assertEqual(track[-1].note, self.note_pitch)
        self.assertEqual(track[-1].time,
                         self.first_midi_note_time + self.ntm.note_duration)
        self.assertEqual(track[-1].type, self.note_off_msg)

    def test_measure_to_messages_counts_time(self):
        """
        Checks that the measure_to_messages function returns the time
        between the last note in the measure and the end of it correctly.
        """
        _, time, _ = self.ntm.measure_to_messages(self.measure, 0, [])
        self.assertEqual(time, self.time_to_measure_end)

    def test_measure_to_messages_counts_active_notes(self):
        """
        Checks that the measure_to_messages function returns the number
        of active notes correctly.
        """
        _, time, _ = self.ntm.measure_to_messages(self.measure, 0, [])
        self.assertEqual(time, self.time_to_measure_end)

    def test_add_meta_messages_prepends_messages(self):
        """
        Checks that the add_meta_messages function prepends the
        track_name and instrument_name meta messages to a midi track correctly.
        """
        track = self.ntm.add_meta_messages(self.track)
        self.assertEqual(len(track), 3)
        self.assertEqual(track[0].type, "track_name")
        self.assertEqual(track[1].type, "instrument_name")

    def test_add_conductor_track(self):
        """
        Checks that the add_conductor_track function adds a track
        to the beginning of a midofile's tracks that contains a 4/4
        time signature message.
        """
        midifile = self.ntm.add_conductor_track(self.midifile)
        self.assertEqual(len(midifile.tracks), 2)
        self.assertEqual(midifile.tracks[0][0].type, "time_signature")
        self.assertEqual(midifile.tracks[0][0].numerator, 4)
        self.assertEqual(midifile.tracks[0][0].denominator, 4)
