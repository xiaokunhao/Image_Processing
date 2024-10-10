import cv2
import os
import numpy as np

def create_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_gaussian_blur(image, ksize=(3, 3), sigma=1.0):
    return cv2.GaussianBlur(image, ksize, sigma)

def laplacian_sharpen(image):
    laplacian_kernel = np.array([
        [1,  1, 1],
        [1, -8, 1],
        [1,  1, 1]
    ])
    laplacian = cv2.filter2D(image, -1, laplacian_kernel)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    laplacian = np.uint8(laplacian)
    sharpened_image = cv2.addWeighted(image, 1.2, laplacian, -0.2, 0)
    return sharpened_image

def unsharp_mask(image, ksize=(3, 3), sigma=1.0, alpha=1.5, beta=-0.5):
    blurred = cv2.GaussianBlur(image, ksize, sigma)
    sharpened_image = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened_image

def adjust_contrast_and_brightness(image, alpha=1.2, beta=20):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def save_image(image, filepath):
    cv2.imwrite(filepath, image)

def main():
    output_directory = 'A3_Result'
    create_output_directory(output_directory)

    image = cv2.imread('A3\\001.3.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # Resize Image (if needed)
    resized_image = cv2.resize(image, (800, int(image.shape[0] * (800 / image.shape[1]))))  # Adjust width to 800 pixels
    display_image(resized_image, 'Resized Image')
    save_image(resized_image, os.path.join(output_directory, 'resized_image.jpg'))

    # Apply Gaussian Blur
    blurred_image = apply_gaussian_blur(resized_image)
    display_image(blurred_image, 'Gaussian Blurred Image')
    save_image(blurred_image, os.path.join(output_directory, 'gaussian_blurred_image.jpg'))

    # Laplacian Sharpened Image after Gaussian Blur
    laplacian_image = laplacian_sharpen(blurred_image)
    display_image(laplacian_image, 'Laplacian Sharpened Image after Gaussian Blur')
    save_image(laplacian_image, os.path.join(output_directory, 'laplacian_sharpened_after_gaussian_blur.jpg'))

    # Unsharp Masked Image after Gaussian Blur
    unsharp_image = unsharp_mask(blurred_image)
    display_image(unsharp_image, 'Unsharp Masked Image after Gaussian Blur')
    save_image(unsharp_image, os.path.join(output_directory, 'unsharp_masked_after_gaussian_blur.jpg'))

    # Adjust contrast and brightness if needed
    final_image = adjust_contrast_and_brightness(unsharp_image)
    display_image(final_image, 'Final Adjusted Image')
    save_image(final_image, os.path.join(output_directory, 'final_adjusted_image.jpg'))

if __name__ == "__main__":
    main()
