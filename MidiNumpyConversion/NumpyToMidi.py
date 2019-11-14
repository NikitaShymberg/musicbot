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
        Initializes the NumpyToMidi class
        """
        self.ppq = PPQ
        # The number of midi ticks per np song tick
        self.pp_time = (1 / NUM_TIMES) * PPQ * 4
        self.note_duration = PPQ

    def numpy_to_midi(self, song: np.ndarray) -> mido.MidiFile:
        """
        Converts the given `song` to a mido MidiFile.

        All notes last for PPQ
        PPQ is set in the constants.py file
        """
        midi_song = mido.MidiFile()
        midi_song = self.add_conductor_track(midi_song)
        midi_song.ticks_per_beat = 960
        midi_track = mido.MidiTrack()
        midi_track = self.add_meta_messages(midi_track)
        cur_time = 0
        on_notes = []
        for measure in song:
            track, cur_time, on_notes = self.measure_to_messages(measure,
                                                                 cur_time,
                                                                 on_notes)
            midi_track += track
        midi_song.tracks.append(midi_track)

        return midi_song

    def add_message(self, track: mido.MidiTrack, note: int, time: int,
                    msg="note_on") -> mido.MidiTrack:
        """
        Appends the message of type `msg` for the given `note` at the given
        `time` to the `track`. Returns the modified track.
        """
        msg = mido.Message(msg, note=note, time=int(time))
        track.append(msg)
        return track

    def measure_to_messages(self, measure: np.ndarray, start_time: int,
                            active_notes: list) -> (mido.MidiTrack, int, list):
        """
        Takes a numpy `measure` and `start_time` (the time between the
        last message in the midifile and the start of this measure)
        and converts it to a miditrack.
        Returns the converted miditrack, the time between the
        last message and the end of the measure, and any notes that
        haven't been turned off as a tuple.
        """
        cur_time = start_time
        track = mido.MidiTrack()
        for note in measure:
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

            cur_time += self.pp_time
            for i in range(len(active_notes)):
                active_notes[i]["time"] += self.pp_time
        return track, cur_time, active_notes

    def add_meta_messages(self, track: mido.MidiTrack) -> mido.MidiTrack:
        """
        Adds meta messages to the beginning of the track including:
        - track name
        - instrument name
        """
        meta_msgs = [
            mido.MetaMessage("track_name", name="Main_Track", time=0),
            mido.MetaMessage("instrument_name", name="Piano", time=0),
                    ]
        track = meta_msgs + track
        return track

    def add_conductor_track(self, song: mido.MidiFile) -> mido.MidiFile:
        """
        Adds track 0 to the given song that adds info about the:
        - time signature
        """
        conductor = mido.MidiTrack()
        conductor.append(mido.MetaMessage("time_signature",
                                          numerator=4, denominator=4))
        song.tracks = [conductor] + song.tracks
        return song


if __name__ == "__main__":
    ntm = NumpyToMidi()
    # song = np.zeros((NUM_MEASURES, NUM_TIMES, NUM_NOTES))
    # song[0, 24, 10] = 1
    # song[0, 24, 11] = 1
    # song = np.load("output/test_epoch_2000.npy").reshape((16, 96, 96)).astype('int8')
    # midi = ntm.numpy_to_midi(song)
    # song = np.unpackbits(np.load("output/test_epoch_2000.npy"),
    #                      axis=-1)
    # midi = ntm.numpy_to_midi(song[0])
    # midi.save("data/temp/bak.mid")

    from Visualization.SongDisplay import SongDisplay
    from train.PrepData import PrepData
    s = np.unpackbits(np.load("data/temp/Amon Amarth - Beheading Of A King.mid.npy"), axis=-1)
    # s = s.reshape((16, 96, 96))
    SongDisplay.show(s[0])
    SongDisplay.show(PrepData.to_song(s[0]))
    s1 = PrepData.to_song(s[0])
    midi = ntm.numpy_to_midi(s1)
    midi.save("data/temp/bak.mid")
