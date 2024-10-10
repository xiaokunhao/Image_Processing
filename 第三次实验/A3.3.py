import cv2
import os

# 输入和输出路径
input_path = "A3/001.3.png"
output_path = "A3.3_Result/001.3_result.png"

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 读取图像
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"无法读取图像：{input_path}")
    exit()

# 应用高斯滤波器进行平滑处理，减少噪声
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# 应用 Canny 边缘检测
low_threshold = 30
high_threshold = 100
edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

# 保存结果图像
cv2.imwrite(output_path, edges)
print(f"边缘检测结果已保存到：{output_path}")
#如果检测结果过于稀疏，则可以降低高阈值，例如设置为80，低阈值相应调整为40。
#如果检测结果包含过多的噪声，可以提高高阈值，例如设置为120，低阈值相应调整为60。