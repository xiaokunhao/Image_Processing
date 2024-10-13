"""
Canny 边缘检测是一种多级边缘检测算法，由 John F. Canny 于 1986 年开发。它被广泛认为是目前最优的边缘检测算法之一，因其良好的检测性能和定位精度而被广泛使用。OpenCV 中有一个非常方便的函数 cv2.Canny() 用于实现 Canny 边缘检测。
Canny 边缘检测的步骤
(1)噪声去除：使用高斯滤波器平滑图像，去除噪声。
(2)计算梯度：使用 Sobel 算子计算图像的梯度强度和方向。
(3)非极大值抑制：去除非边缘点，保留局部最大值作为边缘点。
(4)双阈值法：使用高低两个阈值对边缘进行筛选，确保检测到强边缘，同时连接弱边缘。
(5)边缘连接：将弱边缘点连接到强边缘点。
OpenCV 中的实现
函数原型
cv2.Canny(image, threshold1, threshold2, edges=None, apertureSize=3, L2gradient=False)
参数详解
image：输入图像。应为单通道的灰度图像。
threshold1：低阈值，用于边缘连接的低阈值。
threshold2：高阈值，用于边缘检测的高阈值。
edges：输出图像，与输入图像大小相同，类型为 uint8。
apertureSize：Sobel 算子的孔径大小。默认值为3。
L2gradient：布尔值，指示是否使用更精确的 L2 范数（即两个方向的平方和的平方根）来计算边缘强度。如果为 False，则使用 L1 范数（即两个方向的绝对值和）。
示例代码
"""
import cv2
import matplotlib.pyplot as plt


input_path="Sobel/horse.jpg"
img=cv2.imread(input_path,0)
blurred_image=cv2.GaussianBlur(img,ksize=(5,5),sigmaX=0)
edges=cv2.Canny(blurred_image,50,100)
# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')
plt.show()
