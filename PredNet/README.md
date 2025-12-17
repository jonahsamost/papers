[paper](https://arxiv.org/pdf/1605.08104)

model is composed of a top-down and bottom-up component.

The top down layers (i.e. starting from highest layer down) use
    - current layer's, previous-time-step error
    - current layer's, previus-time-step hidden state
    - next higher layer's, current-time-step hidden state
to generate that layer's current hidden state

The bottom up layers use:
    - current hidden state to generate a prediction
    - a layer-wise "ground truth" thats either the pixels themselves (layer 0) or the propogated error from the previous (lower) level

The goal is to have the top-down layers hidden state be able to perfectly predict the previous layer's residual errors

The model wants to "subtract out" whatever it can predict at each level, and only pass on "surprise" to the next layer
