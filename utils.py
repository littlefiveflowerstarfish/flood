import numpy as np
from PIL import Image
import random

import numpy as np
from scipy.stats import norm

def gaussian_density(n, sigma):
    """
    Generates a NumPy array representing the Gaussian density function.

    Args:
        n (int): Number of values to generate.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: A NumPy array of n values representing the Gaussian density.
    """
    if not isinstance(n, int):
        n = len(n)
    if sigma <= 0:
        raise ValueError("Sigma (standard deviation) must be a positive value.")

    mu = n // 2  # Mean of the Gaussian

    x = np.arange(n) # Create an array of x values from 0 to n-1

    # Gaussian density function formula
    gaussian_array = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    v = gaussian_array[mu]
    return gaussian_array/v

def gaussian_from_onehot(x, sigma, m=1):
    """
    Calculates the probability density of a Gaussian distribution based on a one-hot encoded array.

    Args:
        x: A 1D numpy array representing a one-hot encoded vector.  Only one element
           should be 1, and the rest should be 0.  The position of the '1'
           indicates the mean of the Gaussian.
        sigma: The standard deviation of the Gaussian distribution.

    Returns:
        A 1D numpy array of the same shape as x, containing the probability density
        values of the Gaussian distribution.  Values beyond 3*sigma from the mean
        are set to 0.
    """

    if not isinstance(x, np.ndarray):
        x = x.values
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if np.sum(x) != 1:
        return x*0
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("x must contain only 0s and 1s")
    if not isinstance(sigma, (int, float)):
         raise TypeError("sigma must be a number")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Find the index of the '1' (the mean)
    mean_index = np.where(x == 1)[0][0]

    # Create an array of indices corresponding to the positions in x
    indices = np.arange(len(x))

    # Calculate the Gaussian probability density
    y = norm.pdf(indices, loc=mean_index, scale=sigma)

    # Set values beyond 3*sigma to 0
    distance_from_mean = np.abs(indices - mean_index)
    y[distance_from_mean > 3 * sigma] = 0

    return y*m

def save_array_as_png(array, filename):
    """
    Saves a 3D NumPy array as a PNG file using Pillow (PIL) after normalizing to 0-1.

    Args:
        array: The 3D NumPy array (e.g., shape (128, 128, 3)).
        filename: The path to save the PNG file (e.g., "image.png").
    """
    # Normalize the array to the range 0-1
    array = (array - np.min(array)) / (np.max(array) - np.min(array))

    # Scale to 0-255 and convert to uint8
    array = (array * 255).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(array)

    # Save the image as PNG
    image.save(filename)

def random_flip(image):
    """
    Randomly flip a PIL image horizontally and/or vertically.
    
    Args:
        image (PIL.Image): Input PIL image
        
    Returns:
        PIL.Image: Flipped image
    """
    # Create a copy of the image to avoid modifying the original
    flipped_image = image.copy()

    # Randomly decide whether to flip horizontally (left/right)
    if random.choice([True, False]):
        flipped_image = flipped_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Randomly decide whether to flip vertically (up/down)
    if random.choice([True, False]):
        flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)

    return flipped_image

def constant_logloss(y_true):
    """
    Computes the binary classification logloss when the prediction is constant,
    specifically the average of the true target values.

    Args:
        y_true: A 1D numpy array of true binary labels (0 or 1).

    Returns:
        The logloss value.  Returns infinity if all y_true are the same
        to avoid log(0) error.
    """

    y_true = np.asarray(y_true)  # Ensure it's a numpy array

    if len(y_true) == 0:
        return 0.0  # Handle empty input

    avg_y = np.mean(y_true)

    if avg_y == 0.0 or avg_y == 1.0:  # Edge case: all 0s or all 1s
        return float('inf')

    logloss = -np.mean(y_true * np.log(avg_y) + (1 - y_true) * np.log(1 - avg_y))

    return logloss

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from const import PATH
    tag = 'Train'
    df = pd.read_csv(f'{PATH}/{tag}.csv')
    print(df.head())
    df['order'] = df['event_id'].apply(lambda x: x.split('_')[-1]).astype(int)
    df['location'] = df['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df = df.groupby('location').agg({'label': 'max'}).reset_index()
    print(df.head())
    logloss_value = constant_logloss(df['label'].values)
    print(f"Constant Logloss: {logloss_value}")
