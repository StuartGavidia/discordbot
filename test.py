import numpy as np
from models.GAN import GAN
import os
import matplotlib.pyplot as plt

SECTION = 'gan'
RUN_ID = '0001'
DATA_NAME = 'horse'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

def generate_images(generator, num_images):
    # Generate random noise as input for the generator
    random_latent_vectors = np.random.normal(size=(num_images, gan.z_dim))

    # Generate images from the random noise
    generated_images = generator.predict(random_latent_vectors)

    return generated_images

gan = GAN(input_dim = (28, 28, 3)  # Updated to 3 channels for RGB images
        , discriminator_conv_filters = [64, 64, 128, 128]
        , discriminator_conv_kernel_size = [5, 5, 5, 5]
        , discriminator_conv_strides = [2, 2, 2, 1]
        , discriminator_batch_norm_momentum = None
        , discriminator_activation = 'relu'
        , discriminator_dropout_rate = 0.4
        , discriminator_learning_rate = 0.0008
        , generator_initial_dense_layer_size = (7, 7, 64)
        , generator_upsample = [2, 2, 1, 1]
        , generator_conv_filters = [128, 64, 64, 3]  # Last layer updated to 3 channels
        , generator_conv_kernel_size = [5, 5, 5, 5]
        , generator_conv_strides = [1, 1, 1, 1]
        , generator_batch_norm_momentum = 0.9
        , generator_activation = 'relu'
        , generator_dropout_rate = None
        , generator_learning_rate = 0.0004
        , optimiser = 'rmsprop'
        , z_dim = 100
        )

# Load the trained generator weights
gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# Generate 10 horse images
num_images = 10
generated_images = generate_images(gan.generator, num_images)

# Rescale the generated images to the original range [0, 255]
generated_images = (generated_images + 1) * 127.5
generated_images = np.clip(generated_images, 0, 255).astype('uint8')

# Display the generated images
fig, axs = plt.subplots(1, num_images, figsize=(20, 2))
for i in range(num_images):
    axs[i].imshow(generated_images[i])
    axs[i].axis('off')
plt.show()
