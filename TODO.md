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
1. Migrate the data loader and set it up as a nice class
2. Test the data loader
3. Decide on a deep learning framework (TF2.0/Pytorch)