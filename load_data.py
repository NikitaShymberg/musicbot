import mido
import os
import numpy as np

from utils import time_to_tensor
from constants import NUM_MEASURES, NUM_TIMES, NUM_NOTES

# NOTE: time signature is absolut
note_messages = ["note_on", "note_off"]
all_music = []


def load_midis(path):
    """
    TODO: add a progress bar
    TODO: ignore multi track files? Ignore drums (channel 10)? Handle it
    TODO: NEXTTIME how to play notes at thte same time? Is there a msh in bw?
    """
    for file in os.listdir(path):
        print("Filename:", file)
        midifile = mido.MidiFile(os.path.join(path, file))
        ppq = midifile.ticks_per_beat
        print("PPQ:", ppq)
        messages = mido.merge_tracks(midifile.tracks)
        time_signatures = [m for m in midifile.tracks[0]
                           if m.type == "time_signature"]

        song = np.zeros((NUM_MEASURES, NUM_TIMES, NUM_NOTES))
        time = 0
        cur_time_signature = time_signatures.pop(0)
        cur_measure = 0
        cur_beat = 0
        for note in messages:
            if note.type == "note_on":
                assert note.velocity != 0  # If this never happens we gucci

            time += note.time
            # Update time signature
            if len(time_signatures) != 0 and time > time_signatures[0].time:
                cur_time_signature = time_signatures.pop(0)
                print("NEW TIME SIGNATURE:", cur_time_signature.numerator)

            cur_beat = (time % (cur_time_signature.numerator * ppq)) // ppq
            cur_measure = (time % (cur_time_signature.numerator * ppq)) // ppq

            # Update song
            if note.type == "note_on":
                tensor_time = time_to_tensor(time, ppq)
                print("Time:", time)
                print("Note type:", note.type)
                print("Note time:", note.time)
                print("Note pitch:", note.note)
                print("Beat:", cur_beat)
                print("Measure:", cur_measure)
                print("Tensor time:", tensor_time)
                song[cur_measure, tensor_time, note.note] = 1
                print("-"*88)

        print("Number of songs", len(all_music))
        print("Final time", time)

        # print("Time sigs", [msg for msg in time_signatures])
        # NOTE: note_on with velocity == 0 == note_off
        break
    print("-"*88)
    # Note: ticks per note (ppq) could be 960


if __name__ == "__main__":
    print("Time to tensor:", time_to_tensor(30720, 960))
    load_midis("data")
