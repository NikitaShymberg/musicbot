Project structure
    - Data loader
        - Transform midi to np array
        - Pandas is overkill I think
    - Network
        - Just the decoder
    - PCA
        - Decompose the original dataset
        - Sample from the PCA space and feed into decoder
        - Compare decoder output to the sample from the original dataset
        - Learn
    - Data producer
        - Nparray --> midi
    - Dynamic learning(?)
        - Dynamic network structure
        - Learning rate, batch size, etc.
        - Loss function
        - This just might be a separate project

TODO:
1. Decide on a deep learning framework - TF2.0
og.mid
np.npy
bk1.mid --> ok
bk1.npy --> ok
bk2.mid --> slow
bk2.npy --> slow
Fix bk1.npy --> bk2.mid