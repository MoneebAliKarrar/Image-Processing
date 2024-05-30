
def printPixels(image):
    width, height = image.size
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            print(f"Pixel at ({x}, {y}): R={r}, G={g}, B={b}")