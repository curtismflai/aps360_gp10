# aps360_gp10

Software that enhances images of anime and cartoons using an artificial neural network
.

## (not_so_simple_model.py)
Unused, older code that loads images, pads them, defines an autoencoder, trains it, and outputs the results.

## Final_Model_Training.py (simple_gan.py)
The current code that contains the discriminator and enhancer. This runs a similar process as the previous file. Our current best results are from running the generator only from this file.

## (simple_model.py)
Older code containing a simple autoencoder model.

## validation_script.py (validation.py)
Code used to evaluate loss values each epoch. It can generate loss graphs.

## video_demo_enhancer.py (video_enhancer.py)
Code used to produce upscaled video by upscaling each frame as an image.