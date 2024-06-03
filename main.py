from PIL import Image
import numpy as np
from meanfilter import *
from pixelprint import *
from thresholding import *
from histogramequalization import *

filename = "resources/yoda.jpeg"
img = Image.open(filename)
pixels = img.load()

"""
# Step 1: Compute the histogram and cumulative distribution of the original image
original_hist = img.convert('L').histogram()
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
equalized_image = img.convert('L').point(lambda x: equalized_mapping[x])

# Step 3: Compute the histogram and cumulative distribution of the equalized image
equalized_hist = equalized_image.histogram()
equalized_hist_np = np.array(equalized_hist)

# Compute the cumulative distribution function (CDF)
equalized_cdf = np.cumsum(equalized_hist_np)

# Normalize the CDF
equalized_cdf_normalized = equalized_cdf * equalized_hist_np.max() / equalized_cdf.max()

# Step 4: Plot the results
plt.figure(figsize=(12, 8))

# Original Image and Histogram
plt.subplot(2, 2, 1)
plt.imshow(img.convert('L'), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(original_hist_np, color='black', label='Histogram')
plt.plot(original_cdf_normalized, color='blue', label='Cumulative Distribution')
plt.title('Original Histogram & CDF')
plt.legend()

# Equalized Image and Histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(equalized_hist_np, color='black', label='Histogram')
plt.plot(equalized_cdf_normalized, color='blue', label='Cumulative Distribution')
plt.title('Equalized Histogram & CDF')
plt.legend()

plt.tight_layout()
plt.show()
"""


if __name__ == "__main__":
    """
    #1--
    printPixels(img)
    """
    """
    #2--
    thresh = 50
    image_single_threshold = singlethreshold(img,thresh)

    # Double thresholding
    low_threshold = 50
    high_threshold = 150
    image_double_threshold = doublethreshold(img,low_threshold,high_threshold)

    image_single_threshold.save("yoda_single_threshold.jpg")
    image_double_threshold.save("yoda_double_threshold.jpg")

    image_single_threshold.show()
    image_double_threshold.show()
    showhistogram(img)
    img.close()
    """
    
  
    """
    #3-- histogram equalization
    equalized_image,original_hist_np,original_cdf_normalized,equalized_hist_np,equalized_cdf_normalized=equalizehistogram(img)
    # Plot the results
    plt.figure(figsize=(12, 8))

    # Original Image and Histogram
    plt.subplot(2, 2, 1)
    plt.imshow(img.convert('L'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.plot(original_hist_np, color='black', label='Histogram')
    plt.plot(original_cdf_normalized, color='blue', label='Cumulative Distribution')
    plt.title('Original Histogram & CDF')
    plt.legend()

    # Equalized Image and Histogram
    plt.subplot(2, 2, 3)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.plot(equalized_hist_np, color='black', label='Histogram')
    plt.plot(equalized_cdf_normalized, color='blue', label='Cumulative Distribution')
    plt.title('Equalized Histogram & CDF')
    plt.legend()

    plt.tight_layout()
    plt.show()
    """
    
    
    # Load the grayscale image
    image = Image.open("resources/road.jpg").convert("L")
    # Apply mean filter with a square mask of size 71x71
    #filtered_image = apply_mean_filter(image, mask_size=71)
    filtered_image_naiive = apply_mean_filter_naive(image,mask_size=71)

    # Save the filtered image
    #filtered_image.save("road_mean_filtered.jpg")
    filtered_image_naiive.save("filterd_naiive.jpg")

    # Display the filtered image
    filtered_image_naiive.show()
    