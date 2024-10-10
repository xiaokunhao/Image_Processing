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

def save_image(image, filepath):
    cv2.imwrite(filepath, image)

def main():
    output_directory = 'A3.2_Result'
    create_output_directory(output_directory)

    image_path = 'A3\\001.3-bin.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}.")
        return

    # Display original binary image
    display_image(image, 'Original Binary Image')
    save_image(image, os.path.join(output_directory, 'original_binary_image.jpg'))

    # Define the kernel size
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(image, kernel, iterations=1)
    display_image(eroded_image, 'Eroded Image')
    save_image(eroded_image, os.path.join(output_directory, 'eroded_image.jpg'))

    # Apply dilation
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    display_image(dilated_image, 'Dilated Image')
    save_image(dilated_image, os.path.join(output_directory, 'dilated_image.jpg'))

    # Save the final result
    final_result = dilated_image
    save_image(final_result, os.path.join(output_directory, 'final_result.jpg'))

if __name__ == "__main__":
    main()
