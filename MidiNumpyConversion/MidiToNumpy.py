"""
This file contains the MidiToNumpy class
"""
import mido
import os
import sys
import numpy as np
from tqdm import tqdm

from constants import NUM_TIMES, NUM_MEASURES, NUM_NOTES


class MidiToNumpy():
    """
    Used to convert midi files to numpy arrays.
    Each midi file gets broken up into NUM_MEASURES measure "songs", each song
    has the shape (NUM_MEASURES, NUM_TIMES, NUM_NOTES). If the total number of
    measures in the song is not a multiple of NUM_MEASURES, the end is
    truncated.

    Limitations:
    - This class should only be used for one channel midi files.
    - Time signature meta messages must be in the first track.
    - When a note with a pitch of over 96 comes in, it gets transposed
        an octave down.
    """

    def __init__(self, midi_path: str, numpy_path):
        """
        Creates a MidiToNumpy instance to convert midi files from `midi_path`
        and save them into `numpy_path`.
        """
        self.note_messages = ["note_on", "note_off"]
        self.midi_path = midi_path
        self.numpy_path = numpy_path

    def midi_to_tensor_time(self, time: int, ppq: int,
                            beats_per_measure: int) -> int:
        """
        `time` is the absolute tick that the note is played on
        `ppq` is the pulses per quarter note of the song
        returns the time index in the data tensor.
        """
        return int((time / ppq * (NUM_TIMES / beats_per_measure)) % NUM_TIMES)

    def relative_to_absolute_times(self, notes: [mido.MetaMessage]
                                   ) -> [mido.MetaMessage]:
        """
        Transforms the relative times in `notes` to aboslute times s.t.
        time[i+1] = time[i+1] + time[i].
        """
        for i in range(1, len(notes)):
            notes[i].time += notes[i-1].time
        return notes

    def get_time_signatures(self, midifile: mido.MidiFile
                            ) -> [mido.MetaMessage]:
        """
        Returns all the time signature messsages from the given `midifile`
        in absolute time.

        If no time signatures are found, raises a ValueError.
        """
        abs_times = self.relative_to_absolute_times(midifile.tracks[0])
        time_signatures = [m for m in abs_times if m.type == "time_signature"]
        if len(time_signatures) == 0:
            raise ValueError("ERROR: The given midifile", midifile.filename,
                             "does not have any time signatures")

        return time_signatures

    def midi_file_to_numpy(self, midifile: mido.MidiFile) -> np.ndarray:
        """
        Transforms the given `midifile` into nparray songs. Returns the songs.
        """
        messages = mido.merge_tracks(midifile.tracks)
        ppq = midifile.ticks_per_beat
        time_signatures = self.get_time_signatures(midifile)

        songs = []
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
                    songs.append(song)
                    song = np.zeros((NUM_MEASURES, NUM_TIMES, NUM_NOTES))
                time_of_prev_measure = time

            # Update note
            if note.type == "note_on" and note.velocity != 0:
                tensor_time = self.midi_to_tensor_time(
                    time, ppq, cur_time_signature.numerator)
                while note.note >= 96:
                    note.note -= 12  # Transpose down an octave
                song[cur_measure, tensor_time, note.note] = 1

        return np.asarray(songs)

    def load_midis(self) -> None:
        """
        Loads the midis from `self.midi_path`, coverts them to a numpy format,
        and for each midi file saves a numpy file to the `self.numpy_path`
        directory.
        """
        for file in tqdm(os.listdir(self.midi_path)):
            try:
                midifile = mido.MidiFile(os.path.join(self.midi_path, file))
            except ValueError as e:
                print(e, "Skipping and continuing...", file=sys.stderr)
                continue
            except OSError as e:
                print("ERROR opening", file, "Skipping and continuing...",
                      file=sys.stderr)
                print(e, file=sys.stderr)
                continue

            songs = self.midi_file_to_numpy(midifile)

            # Save every midifile into a separate file to save memory
            np.save(self.numpy_path + file, np.asarray(songs))


if __name__ == "__main__":
    data_loader = MidiToNumpy("data/temp", "data/temp")
    data_loader.load_midis()
