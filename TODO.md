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
3. Mess around with model structure
4. More tests
5. Training hyperparameters and whatever
6. Something to record all testing of model things
7. Detemine conf_thresh
8. Save times