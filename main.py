from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

filename = "resources/yoda.jpeg"
img = Image.open(filename)

pixels = img.load()

"""
##1-printing the pixels
width, height = img.size
for y in range(height):
    for x in range(width):
        r, g, b = pixels[x, y]
        print(f"Pixel at ({x}, {y}): R={r}, G={g}, B={b}")
"""
"""
# Single thresholding
thresh = 50
fnsingle = lambda x : 255 if x > thresh else 0
image_single_threshold = img.convert('L').point(fnsingle, mode='1')

# Double thresholding
low_threshold = 50
high_threshold = 150
fndouble = lambda x: 255 if x > high_threshold else (0 if x < high_threshold else 127)
image_double_threshold = img.convert('L').point(fndouble, mode='1')

image_single_threshold.save("yoda_single_threshold.jpg")
image_double_threshold.save("yoda_double_threshold.jpg")

image_single_threshold.show()
image_double_threshold.show()

image_array = np.array(img.convert('L'))
plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Histogram of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
img.close()
"""


'''
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
'''



def compute_summed_area_table(image):
    """
    Computes the summed-area table for the given grayscale image.
    """
    width, height = image.size
    integral_image = [[0] * width for _ in range(height)]
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            # Pixel value at (x, y)
            pixel_value = pixels[x, y]

            # Above value, if y > 0
            above = integral_image[y - 1][x] if y > 0 else 0

            # Left value, if x > 0
            left = integral_image[y][x - 1] if x > 0 else 0

            # Above-left value, if both x > 0 and y > 0
            above_left = integral_image[y - 1][x - 1] if y > 0 and x > 0 else 0

            # Calculate the integral image value
            integral_image[y][x] = pixel_value + left + above - above_left

    return integral_image
    
def apply_mean_filter(image, mask_size):
    """
    Applies mean filter with the specified mask size to the given image.
    """
    width, height = image.size
    half_mask = mask_size // 2

    integral_image = compute_summed_area_table(image)
    """
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            val = pixels[x,y]
            print(f"Pixel at ({x}, {y}): Val={val}")

    """
    output_image = Image.new("L", (width, height))

    for y in range(height):
        for x in range(width):
            x1 = max(x - half_mask, 0)
            y1 = max(y - half_mask, 0)
            x2 = min(x + half_mask, width - 1)
            y2 = min(y + half_mask, height - 1)
            area_sum = integral_image[y2][x2]
            if x1 > 0:
                area_sum -= integral_image[y2][x1 - 1]
            if y1 > 0:
                area_sum -= integral_image[y1 - 1][x2]
            if x1 > 0 and y1 > 0:
                area_sum += integral_image[y1 - 1][x1 - 1]

            area_size = (x2 - x1 + 1) * (y2 - y1 + 1)
            mean_intensity = area_sum / area_size

            output_image.putpixel((x, y), round(mean_intensity))

    return output_image

# Load the grayscale image
image = Image.open("resources/road.jpg").convert("L")
# Apply mean filter with a square mask of size 71x71
filtered_image = apply_mean_filter(image, mask_size=71)

# Save the filtered image
filtered_image.save("road_mean_filtered.jpg")

# Display the filtered image
filtered_image.show()