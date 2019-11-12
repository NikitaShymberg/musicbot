"""
This file contains all the constants used throughout the project.
"""

# Song constants
NUM_TIMES = 96  # The number of time increments in a single measure
NUM_NOTES = 96  # The number of different note pitches in songs
NUM_MEASURES = 16  # The number of measures in the song
PPQ = 960  # The pulses per quarter note value to use when writing MIDI files

# Learning constants
# The number of dimensions to use in the PCA step
PCA_DIMENSIONS = 2  # TESTING
# PCA_DIMENSIONS = 1000  # Captures ~ 74% variance
# PCA_DIMENSIONS = 1150  # Captures ~ 80% variance
# PCA_DIMENSIONS = 1600  # Captures ~ 90% variance
# PCA_DIMENSIONS = 2000  # Captures ~ 98% variance
EPOCHS = 10
BATCH_SIZE = 16
FLAT_SIZE = NUM_MEASURES * NUM_NOTES * NUM_TIMES
CONF_THRESH = 0.5  # TODO: determine
