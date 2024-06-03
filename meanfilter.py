from PIL import Image

def compute_summed_area_table(image):
    """
    Computes the summed-area table for the given grayscale image.
    """
    width, height = image.size
    integral_image = [[0] * width for _ in range(height)]
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            pixel_value = pixels[x, y]
            above = integral_image[y - 1][x] if y > 0 else 0
            left = integral_image[y][x - 1] if x > 0 else 0
            above_left = integral_image[y - 1][x - 1] if y > 0 and x > 0 else 0
            integral_image[y][x] = pixel_value + left + above - above_left

    return integral_image
    
def apply_mean_filter(image, mask_size):
    """
    Applies mean filter with the specified mask size to the given image.
    """
    width, height = image.size
    half_mask = mask_size // 2

    integral_image = compute_summed_area_table(image)
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

def apply_mean_filter_naive(image, mask_size):
    width, height = image.size
    half_mask = mask_size // 2

    pixels = image.load()
    output_image = Image.new("L", (width, height))

    for y in range(height):
        for x in range(width):
            sum_intensity = 0
            count = 0
            for ky in range(-half_mask, half_mask + 1):
                for kx in range(-half_mask, half_mask + 1):
                    nx, ny = x + kx, y + ky
                    if 0 <= nx < width and 0 <= ny < height:
                        sum_intensity += pixels[nx, ny]
                        count += 1

            mean_intensity = sum_intensity / count
            output_image.putpixel((x, y), round(mean_intensity))

    return output_image