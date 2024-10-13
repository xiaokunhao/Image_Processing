import numpy as  np
import cv2
import matplotlib.pyplot as plt
input_path="Sobel/horse.jpg"
img=cv2.imread(input_path,0)
cv2.imshow('gray',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#使用Sobel算子
#图像的深度指的是每个像素存储颜色信息所使用的位数。它通常表示图像的色彩精细程度和动态范围。图像深度越高，能够表示的颜色或灰度级别就越多
"""
src：输入图像，通常是灰度图像。
ddepth：输出图像的深度。常用值有：
cv2.CV_64F：64位浮点型，适用于需要高精度的计算。
cv2.CV_32F：32位浮点型。
cv2.CV_16S：16位有符号整型。
cv2.CV_8U：8位无符号整型。
dx：x方向的导数阶数。dx=1 表示计算x方向的一阶导数。
dy：y方向的导数阶数。dy=1 表示计算y方向的一阶导数。
ksize：Sobel核的大小，必须为1、3、5、7。默认为3。核越大，检测到的边缘越平滑，但计算量越大。
scale：可选的缩放因子，乘以导数值。用于控制梯度强度。
delta：可选的增量值，会加到结果上。通常用于调整亮度。
borderType：边界模式，定义图像边界的像素外推方法。常用值：
cv2.BORDER_DEFAULT：默认边界模式。
其他边界模式如 cv2.BORDER_CONSTANT、cv2.BORDER_REPLICATE 等
"""
x=cv2.Sobel(img,cv2.CV_16S,1,0,ksize=3)
#cv2.Sobel() 是 OpenCV 中用于图像梯度计算的函数，基于 Sobel 算子。它主要用于边缘检测，计算图像在x和y方向的梯度
y=cv2.Sobel(img,cv2.CV_16S,0,1,ksize=3)
#该段代码将两个图像或矩阵x和y进行绝对值转换并缩放至合适范围（0-255），
# 然后按权重0.5相互叠加合并成新图像res

#cv2.convertScaleAbs() 用于将图像的像素值转换为绝对值并缩放到8位无符号整型（0-255）。
# 这在处理梯度图像时尤其有用，因为梯度计算可能会产生负值，而图像显示必须是正值
absx=cv2.convertScaleAbs(x)
absy=cv2.convertScaleAbs(y)
#cv2.addWeighted() 用于将两幅图像进行加权和。它经常用于图像融合和多通道图像的合成
#将两幅梯度图像加权和，是为了综合水平和垂直方向的梯度信息，得到完整的边缘图像
res=cv2.addWeighted(absx,0.5,absy,0.5,0)
# 使用 matplotlib 显示图像
plt.imshow(res, cmap=plt.cm.gray)
plt.title("Sobel Edge Detection")
plt.axis('off')  # 关闭坐标轴
plt.show()

#使用Schaar
x=cv2.Sobel(img,cv2.CV_16S,1,0,ksize=-1)
y=cv2.Sobel(img,cv2.CV_16S,0,1,ksize=-1)
absx=cv2.convertScaleAbs(x)
absy=cv2.convertScaleAbs(y)
res=cv2.addWeighted(absx,0.5,absy,0.5,0)
cv2.imshow('Schaar',res)
cv2.waitKey(0)
cv2.destroyAllWindows()