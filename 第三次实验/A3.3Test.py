import cv2
import numpy as np
import os

def preprocess_image(image_path):
    result_dir = "A3.3_Result"
    os.makedirs(result_dir, exist_ok=True)
    gray=cv2.imread(image_path,0)
    # 使用自适应阈值化
    blurred_image = cv2.medianBlur(gray, 5)
    binary = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    # 定义结构元素的大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(binary, kernel, iterations=1)
    # 应用开运算
    opened_image = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

    # 存储中间图片
    cv2.imwrite(os.path.join(result_dir, "binary.png"), binary)
    cv2.imwrite(os.path.join(result_dir, "open_image.png"), opened_image)

    return opened_image


def find_contours(preprocessed_img):
    # 使用findContours接口寻找轮廓
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# 主函数
if __name__ == "__main__":
    image_path = 'A3/001.3-bin.png'  # 图像路径
    preprocessed_img = preprocess_image(image_path)
    contours = find_contours(preprocessed_img)
    print(len(contours))
    # 绘制轮廓
    output_img = preprocessed_img.copy()
    # 检查并转换图像通道
    if len(output_img.shape) == 2:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

    # 绘制红色轮廓
    cv2.drawContours(output_img, contours, -1, (0, 0, 255), 3)

    cv2.imwrite('A3/contours2.png', output_img)  # 保存结果