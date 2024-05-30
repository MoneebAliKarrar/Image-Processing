import numpy as np
import matplotlib.pyplot as plt

def equalizehistogram(image):
    # Step 1: Compute the histogram and cumulative distribution of the original image
    original_hist = image.convert('L').histogram()
    original_hist_np = np.array(original_hist)

    # Compute the cumulative distribution function (CDF)
    original_cdf = np.cumsum(original_hist_np)

    # Normalize the CDF
    original_cdf_normalized = original_cdf * original_hist_np.max() / original_cdf.max()

    # Step 2: Apply histogram equalization
    # Create a mapping from the original pixel values to the equalized values
    image_cdf = original_cdf / original_cdf[-1]  # Normalize the CDF to be in range [0, 1]
    equalized_mapping = np.floor(255 * image_cdf).astype('uint8')  # Scale CDF to range [0, 255] and cast to uint8

    # Apply the mapping to the original image
    equalized_image = image.convert('L').point(lambda x: equalized_mapping[x])

    # Step 3: Compute the histogram and cumulative distribution of the equalized image
    equalized_hist = equalized_image.histogram()
    equalized_hist_np = np.array(equalized_hist)

    # Compute the cumulative distribution function (CDF)
    equalized_cdf = np.cumsum(equalized_hist_np)

    # Normalize the CDF
    equalized_cdf_normalized = equalized_cdf * equalized_hist_np.max() / equalized_cdf.max()

    return equalized_image,original_hist_np,original_cdf_normalized,equalized_hist_np,equalized_cdf_normalized