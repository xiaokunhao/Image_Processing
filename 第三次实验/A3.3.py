import cv2
import numpy as np


def preprocess_image(image_path):
    # 读取图像
    img = cv2.imread(image_path, 0)
    img=cv2.medianBlur(img,5)
    # 二值化处理
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 膨胀和腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded


def find_contours(preprocessed_img):
    # 使用findContours接口寻找轮廓
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# 主函数
if __name__ == "__main__":
    image_path = 'A3/001.3-bin.png'  # 图像路径
    preprocessed_img = preprocess_image(image_path)
    contours = find_contours(preprocessed_img)

    # 绘制轮廓
    output_img = preprocessed_img.copy()
    cv2.imshow('lunkuo',output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.drawContours(output_img, contours, -1, (255), 1)
    print(len(contours))
    cv2.imwrite('A3/contours.png', output_img)  # 保存结果