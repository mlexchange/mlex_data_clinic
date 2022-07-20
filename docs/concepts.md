# Concepts

## Training with pytorch_autoencoder
You can train an assortment of neural networks under different conditions according to the
definition of the following parameters:

### Data Augmentation
* Target width: Width in pixels to which the image will be resized.
* Target height: Height in pixels to which the image will be resized.
* Horizontal Flip Probability: Probability of random horizontal flip.
* Vertical Flip Probability: Probability of random vertical flip.
* Brightness: How much to jitter brightness. brightness_factor is chosen uniformly 
from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be 
non negative numbers.
* Contrast: How much to jitter contrast. contrast_factor is chosen uniformly from 
[max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
* Saturation: How much to jitter saturation. saturation_factor is chosen uniformly from 
[max(0, 1 - saturation), 1 + saturation]  or the given [min, max]. Should be non negative numbers.
* Hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. 
* Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. To jitter hue, the pixel values of the input 
* image has to be non-negative for conversion to HSV space; thus it does not work if you normalize your 
* image to an interval with negative values, or use an interpolation that generates negative values before 
* using this function.

### Training setup
* Latent Dimension: Dimension size of latent space (Lx1).
* Shuffle: Shuffle dataset.
* Batch Size: The number of images in a batch.
* Validation Percentage: Percentage of training images that should be used for validation.
* Base Channel Size: Size of the base channel in the autoencoder.
* Depth: Number of instances where the image size is decreased and the number of channels is
increased per side in the network architecture.
* Optimizer: A specific implementation of the gradient descent algorithm.
* Criterion.
* Learning Rate: A scalar used to train a model via gradient descent.
* Number of Epochs: An epoch is a full training pass over the entire dataset such that 
each image has been seen once.
* Seed: Initialization reference for the pseudo-random number generator. Set up this value 
for the reproduction of the results.

## Output
The output of the training step is the trained model.

## Prediction with pytorch_autoencoder
To predict the reconstructed images of a given testing dataset, you can define the following 
parameters:

### Testing setup
* Batch Size: The number of images in a batch.
* Seed: Initialization reference for the pseudo-random number generator. Set up this value 
for the reproduction of the results.

Similarly to the training step, this approach will resize your dataset to the previously selected
target width and height.

## Output
The output of the prediction step is `dist_matrix.csv` file with the distance matrix between the
testing images, and their corresponding reconstructed images at the output of the autoencoder.
