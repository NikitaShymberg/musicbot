"""
This file contains the DataLoader class
TODO: usage or something
"""
import mido
import os
import numpy as np
from tqdm import tqdm

from constants import NUM_TIMES, NUM_MEASURES, NUM_NOTES


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

    def midi_to_tensor_time(self, time: int, ppq: int,
                            beats_per_measure: int) -> int:
        """
        `time` is the absolute tick that the note is played on
        `ppq` is the pulses per quarter note of the song
        returns the time index in the data tensor
        """
        return int((time / ppq * (NUM_TIMES / beats_per_measure)) % NUM_TIMES)

    def relative_to_absolute_times(self, notes: [mido.MetaMessage]
                                   ) -> [mido.MetaMessage]:
        """
        Transforms the relative times in `notes` to aboslute times s.t.
        time[i+1] = time[i+1] + time[i]
        """
        for i in range(1, len(notes)):
            notes[i].time += notes[i-1].time
        return notes

    def load_midis(self) -> np.ndarray:
        """
        Loads the midis from self.path and returns them as an nparray
        TODO: add a progress bar
        TODO: ignore multi track files? Ignore drums (channel 10)? Handle it
        """
        all_songs = []
        for file in tqdm(os.listdir(self.path)):
            try:
                midifile = mido.MidiFile(os.path.join(self.path, file))
                messages = mido.merge_tracks(midifile.tracks)
            except Exception as e:
                print("ERROR OPENING", file)
                print(e)
                continue

            ppq = midifile.ticks_per_beat
            time_signatures = [m for m in midifile.tracks[0]
                               if m.type == "time_signature"]
            if len(time_signatures) == 0:
                print("ERROR", file, "Doesn't have a time signature")
                continue

            time_signatures = self.relative_to_absolute_times(time_signatures)

            song = np.zeros((NUM_MEASURES, NUM_TIMES, NUM_NOTES))
            time = 0
            cur_time_signature = time_signatures.pop(0)
            cur_measure = 0
            time_of_prev_measure = 0
            for note in messages:
                time += note.time
                # Update time signature
                if len(time_signatures) and time > time_signatures[0].time:
                    cur_time_signature = time_signatures.pop(0)

                measure_duration = cur_time_signature.numerator * ppq
                if time >= time_of_prev_measure + measure_duration:
                    # Update current measure
                    time_since_prev = time - time_of_prev_measure
                    cur_measure += time_since_prev // measure_duration
                    if cur_measure >= NUM_MEASURES:
                        # Update current song
                        cur_measure %= NUM_MEASURES
                        all_songs.append(song)
                        song = np.zeros((NUM_MEASURES, NUM_TIMES, NUM_NOTES))
                    time_of_prev_measure = time

                # Update note
                if note.type == "note_on" and note.velocity != 0:
                    tensor_time = self.midi_to_tensor_time(
                        time, ppq, cur_time_signature.numerator)
                    while note.note >= 96:
                        note.note -= 12
                    song[cur_measure, tensor_time, note.note] = 1

            # Save every song into a separate file to save memory
            np.save("./data/np/" + file, np.asarray(all_songs))
            all_songs = []

        print("Number of songs", len(all_songs))
        return np.asarray(all_songs)
        # Note: ticks per note (ppq) could be 960


if __name__ == "__main__":
    data_loader = DataLoader("data/midi")
    data = data_loader.load_midis()
    # np.save("./midis", data)
