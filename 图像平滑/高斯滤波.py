import cv2
import os

if __name__ == "__main__":
    input_path = 'Le.png'
    image = cv2.imread(input_path, 0)

    if image is not None:
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)

        # 创建结果目录
        result_dir = "Results"
        os.makedirs(result_dir, exist_ok=True)

        # 测试不同大小的高斯滤波核
        for kernel_size in range(3, 12, 2):  # 例如，测试3x3到11x11的奇数核
            blurred = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)  # sigmaX 设置为 0
            output_path = os.path.join(result_dir, f'GaussianBlurred_{kernel_size}x{kernel_size}.png')
            cv2.imwrite(output_path, blurred)
            print(f"保存滤波结果于: {output_path}")

    else:
        print("无法打开图像，请检查路径和文件完整性")
