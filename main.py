import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def rgb_to_lms(rgb):
    rgb_clipped = np.clip(rgb, 0, 255)
    lms_matrix = np.array([[17.8824, 43.5161, 4.11935],
                           [3.45565, 27.1554, 3.86714],
                           [0.0299566, 0.184309, 1.46709]])
    lms = np.dot(rgb_clipped, lms_matrix.T)
    return lms

def simulate_color_blindness(lms, type='Protanopia'):
    if type == 'Protanopia':
        transform_matrix = np.array([[0, 2.02344, -2.52581],
                                     [0, 1, 0],
                                     [0, 0, 1]])
    elif type == 'Deuteranopia':
        transform_matrix = np.array([[1, 0, 0],
                                     [0.494207, 0, 1.24827],
                                     [0, 0, 1]])
    elif type == 'Tritanopia':
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [-0.395913, 0.801109, 0]])
    else:
        raise ValueError("Unsupported color blindness type")
    simulated_lms = np.dot(lms, transform_matrix.T)
    return simulated_lms

def lms_to_rgb(lms):
    rgb_matrix = np.array([[0.080944, -0.130504, 0.116721],
                           [-0.0102485, 0.0540194, -0.113615],
                           [-0.000365294, -0.00412163, 0.693513]])
    rgb = np.dot(lms, rgb_matrix.T)
    rgb_clipped = np.clip(rgb, 0, 255)
    return rgb_clipped.astype(np.uint8)

def lms_daltonization(rgb_image, type='Protanopia', rate=1):
    lms_image = rgb_to_lms(rgb_image)
    simulated_lms = simulate_color_blindness(lms_image, type)
    simulated_rgb = lms_to_rgb(simulated_lms)
    return simulated_rgb

def enhance_saturation(rgb_image, saturation_factor=1.9):
    lab_image = color.rgb2lab(rgb_image)
    lab_image[:, :, 1] *= saturation_factor
    enhanced_rgb_image = color.lab2rgb(lab_image) * 255
    enhanced_rgb_clipped = np.clip(enhanced_rgb_image, 0, 255)
    return enhanced_rgb_clipped.astype(np.uint8)

def plot_fourier_spectrum(image_path, threshold=100):
    # Load the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Compute the Fourier Transform and shift
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)

    # Calculate the magnitude spectrum (log scale for visualization)
    magnitude_spectrum = np.abs(f_shift)
    log_magnitude_spectrum = np.log(1 + magnitude_spectrum)

    # Apply frequency-based filtering
    filtered_spectrum = magnitude_spectrum * (magnitude_spectrum > threshold)

    # Inverse Fourier Transform to restore the image
    f_shift_filtered = f_shift * (magnitude_spectrum > threshold)
    image_restored = np.abs(ifft2(ifftshift(f_shift_filtered)))

    # Display the magnitude spectrum and restored image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(log_magnitude_spectrum, cmap='gray')
    plt.title('Log Magnitude Spectrum')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_spectrum, cmap='gray')
    plt.title('Filtered Spectrum (Threshold = {})'.format(threshold))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_restored, cmap='gray')
    plt.title('Restored Image')
    plt.axis('off')

    plt.show()

def process_image(image_path, blindness_type='Protanopia', rate=5, saturation_factor=4):
    # Open the image using PIL
    img = Image.open(image_path)
    rgb_image = np.array(img)

    # Perform color blindness simulation
    simulated_rgb = lms_daltonization(rgb_image, type=blindness_type, rate=rate)

    # Enhance saturation
    saturated_rgb_image = enhance_saturation(simulated_rgb, saturation_factor=saturation_factor)

    # Save the processed image
    processed_image = Image.fromarray(saturated_rgb_image)
    processed_image.save('processed_image.jpg')

if __name__ == '__main__':
    # Prompt user for input
    input_image_path = input("Enter path to input image: ")
    blindness_type = input("Enter type of color blindness (Protanopia/Deuteranopia/Tritanopia): ")

    # Validate color blindness type
    if blindness_type not in ['Protanopia', 'Deuteranopia', 'Tritanopia']:
        print("Invalid color blindness type. Supported types are Protanopia, Deuteranopia, Tritanopia.")
        exit(1)

    # Process the image
    process_image(input_image_path, blindness_type)
    plot_fourier_spectrum(input_image_path, threshold=100)
