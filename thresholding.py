import matplotlib.pyplot as plt
import numpy as np
def singlethreshold(image,thresh):
    fnsingle = lambda x : 255 if x > thresh else 0
    image_single_threshold = image.convert('L').point(fnsingle, mode='1')
    return image_single_threshold

# Double thresholding
def doublethreshold(image,low_threshold,high_threshold):
    fndouble = lambda x: 255 if x > high_threshold else (0 if x > low_threshold else 127)
    image_double_threshold = image.convert('L').point(fndouble, mode='1')
    return image_double_threshold

def showhistogram(image):
    image_array = np.array(image.convert('L'))
    plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()
    ##image.close()