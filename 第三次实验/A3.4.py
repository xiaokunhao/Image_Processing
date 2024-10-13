import cv2
import numpy as np


def create_mask(image_path, contours):
    # 创建与原图相同尺寸的掩码
    mask = np.zeros_like(cv2.imread(image_path, 0))
    cv2.imshow('mask1', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 确保轮廓数据为正确的格式和类型
    if isinstance(contours, list):
        contours = [c.astype(np.int32) if c.dtype != np.int32 else c for c in contours]
        contours = [c.reshape(-1, 1, 2) if c.shape[1] != 1 else c for c in contours]

    # 填充轮廓
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    cv2.imshow('mask2',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mask


def highlight_regions(image_path, mask):
    # 读取原图
    img = cv2.imread(image_path, 0)
    # 高亮文本区域
    highlighted = cv2.bitwise_and(img,mask)
    return highlighted


# 主函数
if __name__ == "__main__":
    image_path = 'A3/001.3.png'  # 原图像路径
    contours_path = 'A3/contours2.png'  # 轮廓图像路径

    # 读取轮廓图像并转换为灰度图
    contours_image = cv2.imread(contours_path, 0)

    # 使用阈值处理来提取轮廓
    _, thresholded = cv2.threshold(contours_image, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建掩码
    mask = create_mask(image_path, contours)

    # 高亮区域
    highlighted_img = highlight_regions(image_path, mask)
    cv2.imwrite('A3/highlighted.png', highlighted_img)  # 保存结果
