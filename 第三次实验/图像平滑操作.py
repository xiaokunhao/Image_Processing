import cv2
from matplotlib import pyplot as plt

def display_image(image, title, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # 读取图像
    image = cv2.imread('A3\\001.3.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # 原始图像
    display_image(image, 'Original Image')

    # 均值模糊
    mean_blurred = cv2.blur(image, (3,3 ))
    display_image(mean_blurred, 'Mean Blurred')

    # 高斯模糊
    gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    display_image(gaussian_blurred, 'Gaussian Blurred')

    # 中值模糊
    median_blurred = cv2.medianBlur(image, 5)
    display_image(median_blurred, 'Median Blurred')

    # 双边滤波
    bilateral_blurred = cv2.bilateralFilter(image, 9, 75, 75)
    display_image(bilateral_blurred, 'Bilateral Blurred')

if __name__ == "__main__":
    main()
