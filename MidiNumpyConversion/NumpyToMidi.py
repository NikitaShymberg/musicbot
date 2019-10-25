"""
This file contains the NumpyToMidi class
"""
import mido
import numpy as np

from constants import PPQ, NUM_TIMES, NUM_MEASURES, NUM_NOTES


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
        self.pp_time = (1 / NUM_TIMES) * PPQ * 4
        self.note_duration = PPQ

    def numpy_to_midi(self, song: np.ndarray) -> mido.MidiFile:
        """
        TODO: implement
        """
        midi_song = mido.MidiFile()
        midi_song = self.add_conductor_track(midi_song)
        midi_track = mido.MidiTrack()
        cur_time = 0
        for measure in song:
            new_track, new_time = self.measure_to_messages(measure, cur_time)
            midi_track += new_track
            cur_time = new_time
        midi_song.tracks.append(midi_track)

        return midi_song

    def add_message(self, track: mido.MidiTrack, note: int, time: int,
                    msg="note_on") -> mido.MidiTrack:
        """
        Appends the message of type `msg` for the given `note` at the given
        `time` to the `track`. Returns the modified track.
        """
        msg = mido.Message(msg, note=note, time=time)
        track.append(msg)
        return track

    def measure_to_messages(self, measure: np.ndarray, start_time: int
                            ) -> (mido.MidiTrack, int):
        """
        Takes a numpy `measure` and `start_time` (the time between the
        last message in the midifile and the start of this measure)
        and converts it to a miditrack.
        Returns the converted miditrack and the time between the
        last message and the end of the measure as a tuple.
        """
        cur_time = start_time
        active_notes = []
        track = mido.MidiTrack()
        for note in measure:
            if (note == 0).all():
                cur_time += self.pp_time
                for i in range(len(active_notes)):
                    active_notes[i]["time"] += self.pp_time
            else:
                pitches = np.where(note == 1)[0]
                for pitch in pitches:
                    track = self.add_message(track, pitch, cur_time)
                    cur_time = 0
                    active_notes.append({"pitch": pitch, "time": 0})

            for i, active_note in enumerate(active_notes):
                if active_note["time"] >= self.note_duration:
                    track = self.add_message(track, active_note["pitch"],
                                             cur_time, msg="note_off")
                    active_notes[i] = None
                    cur_time = 0
            active_notes[:] = [n for n in active_notes if n is not None]

        return track, cur_time

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


if __name__ == "__main__":
    ntm = NumpyToMidi()
    song = np.zeros((NUM_MEASURES, NUM_TIMES, NUM_NOTES))
    song[0, 24, 10] = 1
    song[0, 24, 11] = 1
    ntm.numpy_to_midi(song)
