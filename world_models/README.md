### World models
- https://arxiv.org/pdf/1803.10122


This model is essentially three models. 
1. VAE
    - used to reconstruct images 
2. MDN-RNN
    - used to predict the next latent vector from the VAE given the last latent representation, previous hidden state of the RNN, and the action taken
3. Controller
    - Linear model that tries to map latent vector and hidden state to action to take

The MDN-RNN is meant to be the "world model" that is able to understand some environment