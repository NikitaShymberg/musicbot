"""
This file contains the NumpyToMidi class
"""
import mido
import numpy as np

from constants import PPQ, NUM_TIMES


class NumpyToMidi():
    """
    Used to convert numpy songs into midi files.
    """
    def __init__(self):
        """
        TODO: implement
        """
        self.ppq = PPQ
        # The number of midi ticks per np song tick
        self.pp_time = (1 / NUM_TIMES) * PPQ
        self.note_duration = PPQ

    def numpy_to_midi(self, song: np.ndarray) -> mido.MidiFile:
        """
        TODO: implement
        """
        midi_song = mido.MidiFile()
        midi_song = self.add_conductor_track(midi_song)
        midi_track = mido.MidiTrack()
        for measure in song:
            midi_track += self.measure_to_messages(measure)

        return midi_song

    def measure_to_messages(self, measure: np.ndarray) -> mido.MidiTrack:
        """
        TODO: implement
        """
        cur_time = 0
        active_notes = []
        for note in measure:
            if (note == 0).all():
                cur_time += self.pp_time
                # TODO: update active notes
            else:
                midi_note = mido.Message("note_on")

            # TODO: turn off active notes that have been on for too long

    def add_meta_messages(self, track: mido.MidiTrack) -> mido.MidiTrack:
        """
        Adds meta messages to the beginning of the track including:
        - track name
        - TODO ???
        """
        ...

    def add_conductor_track(self, song: mido.MidiFile) -> mido.MidiFile:
        """
        Adds track 0 to the given song that adds info about the:
        - time signature
        - tempo
        - ppq(?)
        """
        ...
