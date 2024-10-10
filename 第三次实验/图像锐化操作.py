import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def create_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def display_image(image, title, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def laplacian_sharpen(image):
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])
    laplacian = cv2.filter2D(image, -1, laplacian_kernel)
    sharpened_image = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)
    return sharpened_image


def unsharp_mask(image, ksize=(5, 5), sigma=1.0, alpha=1.5, beta=-0.5):
    blurred = cv2.GaussianBlur(image, ksize, sigma)
    sharpened_image = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened_image


def high_boost_filter(image, ksize=(5, 5), sigma=1.0, k=1.5):
    blurred = cv2.GaussianBlur(image, ksize, sigma)
    sharpened_image = cv2.addWeighted(image, 1 + k, blurred, -k, 0)
    return sharpened_image


def save_image(image, filepath):
    cv2.imwrite(filepath, image)


def main():
    output_directory = 'ruihua'
    create_output_directory(output_directory)

    image = cv2.imread('A3\\001.3.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # Original Image
    display_image(image, 'Original Image')
    save_image(image, os.path.join(output_directory, 'original_image.jpg'))

    # Laplacian Sharpened Image
    laplacian_image = laplacian_sharpen(image)
    display_image(laplacian_image, 'Laplacian Sharpened Image')
    save_image(laplacian_image, os.path.join(output_directory, 'laplacian_sharpened_image.jpg'))

    # Unsharp Masked Image
    unsharp_image = unsharp_mask(image)
    display_image(unsharp_image, 'Unsharp Masked Image')
    save_image(unsharp_image, os.path.join(output_directory, 'unsharp_masked_image.jpg'))

    # High-Boost Filtered Image
    high_boost_image = high_boost_filter(image)
    display_image(high_boost_image, 'High-Boost Filtered Image')
    save_image(high_boost_image, os.path.join(output_directory, 'high_boost_filtered_image.jpg'))


if __name__ == "__main__":
    main()
