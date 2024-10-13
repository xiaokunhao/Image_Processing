import cv2
import numpy as np
import os

# 读取图像并转换为灰度图像
input_path = "horse.jpg"
color_image = cv2.imread(input_path)

if color_image is None:
    print(f"Error: Unable to open image file at {input_path}")
else:
    # 确保 "Results" 目录存在
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 定义拉普拉斯算子卷积核
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

    # 使用拉普拉斯算子计算图像的二阶导数
    """
     cv2.filter2D() 是 OpenCV 中用于对图像进行卷积操作的函数。它可以应用自定义的卷积核来对图像进行滤波
     """
    laplacian = cv2.filter2D(gray_image, cv2.CV_64F, laplacian_kernel)
    laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)

    # 将结果加回原始灰度图像，得到锐化后的灰度图像
    """
     设置权重为1和-1的原因如下：
     权重1：保留原始图像的信息。
     权重-1：对拉普拉斯算子处理后的图像取反，使其与原始图像叠加时突出边缘细节。
     结果：增强图像边缘，达到锐化效果
     """
    sharp_gray_image = cv2.addWeighted(gray_image, 1, laplacian, -1, 0)

    # 将锐化后的灰度图像转换回彩色图像
    sharp_color_image = cv2.cvtColor(sharp_gray_image, cv2.COLOR_GRAY2BGR)
    """
    不直接使用锐化后的灰度图像转换回的彩色图像的原因在于：
       保留色彩信息：直接转换可能会导致色彩失真或不自然。
       平衡效果：通过融合原彩色图像和锐化后的图像，可以更好地保留原有的色彩信息，同时增强图像的细节和清晰度
    """
    # 替换彩色原图中的亮度通道，保留色彩信息
    sharp_color_image = cv2.addWeighted(color_image, 0.5, sharp_color_image, 0.5, 0)

    # 保存原始图像和锐化后的图像
    cv2.imwrite(os.path.join(output_dir, "original_horse.jpg"), color_image)
    cv2.imwrite(os.path.join(output_dir, "sharp_horse.jpg"), sharp_color_image)

    # 显示结果
    cv2.imshow('Sharp Color Image', sharp_color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







