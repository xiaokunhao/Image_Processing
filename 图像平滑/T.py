import cv2
import numpy as np

# 读取灰度图像
image_path = "Le.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 应用二值化处理
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 可视化检查
cv2.imshow("Original Image", image)
cv2.imshow("Binary Image", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 统计指标
foreground_pixels = np.sum(binary_image == 255)
background_pixels = np.sum(binary_image == 0)
total_pixels = binary_image.size
foreground_ratio = foreground_pixels / total_pixels
background_ratio = background_pixels / total_pixels

print(f"Foreground pixels: {foreground_pixels}")
print(f"Background pixels: {background_pixels}")
print(f"Foreground/Total ratio: {foreground_ratio:.2f}")
print(f"Background/Total ratio: {background_ratio:.2f}")

# 应用边缘检测
edges = cv2.Canny(binary_image, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contours", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
