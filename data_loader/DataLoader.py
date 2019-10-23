"""
This file contains the DataLoader class
TODO: usage or something
"""
import mido
import os
import numpy as np

from constants import NOTES_PER_BEAT, NUM_TIMES, NUM_MEASURES, NUM_NOTES


class DataLoader():
    """
    TODO: doc
    """

    def __init__(self, path: str):
        """
        Creates a DataLoader instance to load data from `path`
        """
        self.note_messages = ["note_on", "note_off"]
        self.path = path

    def midi_to_tensor_time(self, time: int, ppq: int, beats_per_measure: int):
        """
        `time` is the absolute tick that the note is played on
        `ppq` is the pulses per quarter note of the song
        returns the time index in the data tensor
        """
        return int((time / ppq * (NUM_TIMES / beats_per_measure)) % NUM_TIMES)

    def relative_to_absolute_times(self, notes: [mido.MetaMessage]):
        """
        Transforms the relative times in `notes` to aboslute times s.t.
        time[i+1] = time[i+1] + time[i]
        """
        for i in range(1, len(notes)):
            notes[i].time += notes[i-1].time
        return notes

    def load_midis(self):
        """
        TODO: add a progress bar
        TODO: ignore multi track files? Ignore drums (channel 10)? Handle it
        """
        all_songs = []
        for file in os.listdir(self.path):
            print("Filename:", file)
            midifile = mido.MidiFile(os.path.join(self.path, file))
            ppq = midifile.ticks_per_beat
            print("PPQ:", ppq)
            messages = mido.merge_tracks(midifile.tracks)
            time_signatures = [m for m in midifile.tracks[0]
                               if m.type == "time_signature"]
            time_signatures = self.relative_to_absolute_times(time_signatures)

            song = np.zeros((NUM_MEASURES, NUM_TIMES, NUM_NOTES))
            time = 0
            cur_time_signature = time_signatures.pop(0)
            cur_measure = 0
            cur_beat = 0
            time_of_prev_measure = 0
            for note in messages:
                if note.type == "note_on":
                    assert note.velocity != 0  # Handle this if it happens

                time += note.time
                # Update time signature
                if len(time_signatures) and time > time_signatures[0].time:
                    cur_time_signature = time_signatures.pop(0)
                    print("NEW TIME SIGNATURE:", cur_time_signature.numerator)

                measure_duration = cur_time_signature.numerator * ppq
                if time >= time_of_prev_measure + measure_duration:
                    time_since_prev = time - time_of_prev_measure
                    cur_measure = (cur_measure + time_since_prev // measure_duration) % NUM_MEASURES
                    time_of_prev_measure = time
                # TODO: do I need the beat??
                cur_beat = (time % (measure_duration)) // ppq
                # song_duration = measure_duration * NUM_MEASURES
                # cur_measure = time % song_duration // (measure_duration)

                # Update song
                if note.type == "note_on":
                    tensor_time = self.midi_to_tensor_time(
                        time, ppq, cur_time_signature.numerator)
                    print("Time:", time)
                    print("Note type:", note.type)
                    print("Note time:", note.time)
                    print("Note pitch:", note.note)
                    print("Beat:", cur_beat)
                    print("Measure:", cur_measure)
                    print("Tensor time:", tensor_time)
                    song[cur_measure, tensor_time, note.note] = 1
                    print("-"*88)

            # print("Number of songs", len(all_music))
            print("Final time", time)

            # print("Time sigs", [msg for msg in time_signatures])
            break
        print("-"*88)
        # Note: ticks per note (ppq) could be 960


if __name__ == "__main__":
    data_loader = DataLoader("data")
    data_loader.load_midis()
