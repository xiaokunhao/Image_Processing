import cv2
import numpy as np
import os

# 输入和输出路径
input_path = "A3/001.3-bin.png"
output_path = "A3.3_Result/001.3_result.png"

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 读取二值化图像
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"无法读取图像：{input_path}")
    exit()

# 逆转二值图像，使文本区域为白色（255），背景为黑色（0）
image = cv2.bitwise_not(image)

# 应用中值滤波平滑图像
smoothed = cv2.medianBlur(image, 11)

# 应用膨胀操作连接文本行形成大型连通域
dilate_kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(smoothed, dilate_kernel, iterations=3)

# 查找轮廓
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制轮廓
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# 保存结果图像
cv2.imwrite(output_path, result_image)
print(f"轮廓检测结果已保存到：{output_path}")
