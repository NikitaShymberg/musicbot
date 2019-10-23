from constants import NOTES_PER_BEAT, NUM_TIMES


def note_to_tensor(note):
    """
    TODO: implement
    """
    pass


def time_to_tensor(time, ppq):
    """
    `time` is the absolute tick that the note is played on
    `ppq` is the pulses per quarter note of the song
    returns the time index in the data tensor
    TODO: handle floats
    """
    return int((time / ppq * NOTES_PER_BEAT) % NUM_TIMES)
    # return int((time / ppq * NOTES_PER_BEAT * beats_per_measure) % NUM_TIMES)
    # return beat * NUM_TIMES / beats_per_measure + time / ppq * NOTES_PER_BEAT


def tensor_to_time():
    """TODO: implement"""
    pass
