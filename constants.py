NUM_TIMES = 96
NUM_NOTES = 96
NUM_MEASURES = 16
PPQ = 960
PCA_DIMENSIONS = 1000  # Captures ~ 74% variance
# PCA_DIMENSIONS = 1150  # Captures ~ 80% variance
# PCA_DIMENSIONS = 1600  # Captures ~ 90% variance
# PCA_DIMENSIONS = 2000  # Captures ~ 98% variance
# PCA_DIMENSIONS = 2  # TESTING

EPOCHS = 10
BATCH_SIZE = 16
FLAT_SIZE = NUM_MEASURES * NUM_NOTES * NUM_TIMES
