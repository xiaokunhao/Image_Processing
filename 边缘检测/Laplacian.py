import numpy as  np
import cv2
import matplotlib.pyplot as plt

from 第三次实验.图像锐化操作 import laplacian_sharpen

"""
#拉普拉斯算子（Laplacian Operator）是一种用于图像边缘检测的二阶导数算子。它通过计算图像的二阶导数来检测图像中的边缘。
# 拉普拉斯算子对噪声比较敏感，因此通常在使用前会对图像进行一些预处理，如高斯平滑。
"""

"""
cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
参数详解
src：输入图像。通常是灰度图像，但也可以是彩色图像。
ddepth：输出图像的深度。常用值包括 cv2.CV_8U、cv2.CV_16U、cv2.CV_32F 和 cv2.CV_64F 等。通常选择较高的深度（如 cv2.CV_64F）以避免溢出，然后再将结果转换回 8 位图像。
ksize：可选参数，表示拉普拉斯算子的核大小。必须为奇数（1, 3, 5, 7）。默认值为 1，表示使用 3x3 的核。
scale：可选参数，表示缩放因子，用于缩放导数值。默认值为 1。
delta：可选参数，表示一个可选的增量值，会加到结果上。默认值为 0。
borderType：可选参数，表示边界模式，定义图像边界的像素外推方法。默认值为 cv2.BORDER_DEFAULT。常用的值包括：
cv2.BORDER_CONSTANT：常数边界，填充一个常数值。
cv2.BORDER_REPLICATE：复制边界的像素值。
cv2.BORDER_REFLECT：镜像边界，镜像翻转边界内的像素。
"""


input_path="Sobel/horse.jpg"
img=cv2.imread(input_path,0)
cv2.imshow('gray',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#先使用高斯模糊以减少噪声
blurred_image = cv2.GaussianBlur(img, ksize=(3, 3),sigmaX=1)
#计算拉普拉斯算子
laplacian=cv2.Laplacian(blurred_image,cv2.CV_64F,ksize=3)
#转换为8位图像
abs_laplacion=cv2.convertScaleAbs(laplacian)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Laplacian')
plt.imshow(abs_laplacion, cmap='gray')

plt.show()



