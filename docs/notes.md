# Prior-art + Strategies

# General Architecture
We will build infrastructure that takes in a set of images and assigns up to 5 whale IDs to each image.  The decision code is what we might call a "model".  We will want the ability to swap out different models to compare evaluated submissions.

## Libraries

- tensorflow
- pytorch
- opencv: Image manipulation library

## Pretrained Models

# Image Issues

- Out of focus
- Non-relevant borders
- Rotated with whitespace borders
- Very partial tails
- Inconsistent or stretched out resolutions