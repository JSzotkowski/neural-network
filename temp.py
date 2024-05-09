import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import load_data_wrapper
import random

def display_digits(images, labels, num_images=5):
    """
    Display digits from the MNIST dataset.

    Args:
        images: List of image data (numpy arrays).
        labels: List of corresponding labels (integers).
        num_images: Number of images to display (default is 5).
    """
    fig, axes = plt.subplots(1, num_images, figsize=(10, 3))
    for i in range(num_images):
        axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.show()


# Load MNIST data
training_data, _, _ = load_data_wrapper()
random.shuffle(training_data)
images, labels = zip(*training_data)

# Display some digits
display_digits(images, labels)
