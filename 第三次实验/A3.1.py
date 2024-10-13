import cv2
import os
import numpy as np

# 定义字体路径（这里以常见的雅黑体为例）
font_path = 'simhei.ttf'  # 或者其他支持中文的字体文件路径
font = cv2.FONT_ITALIC
font_scale = 1
font_color = (255, 255, 255)
line_type = 2
def apply_blur(image, method='mean', **kwargs):
    """
    对图像应用不同类型的模糊。

    参数:
    - image: 输入图像。
    - method: 模糊方法。选项有 'mean', 'gaussian', 'median', 'bilateral'。
    - kwargs: 特定模糊方法的附加参数。

    返回:
    - 模糊后的图像。
    """
    if method == 'mean':
        kernel_size = kwargs.get('kernel_size', (3, 3))
        return cv2.blur(image, kernel_size)

    elif method == 'gaussian':
        kernel_size = kwargs.get('kernel_size', (3, 3))
        sigma = kwargs.get('sigma', 0)
        return cv2.GaussianBlur(image, kernel_size, sigma)

    elif method == 'median':
        kernel_size = kwargs.get('kernel_size', 3)
        return cv2.medianBlur(image, kernel_size)

    elif method == 'bilateral':
        diameter = kwargs.get('diameter', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    else:
        raise ValueError("无效的方法。请选择 'mean', 'gaussian', 'median', 'bilateral' 之一。")


def apply_sharpen(image):
    """
    对图像应用锐化。

    参数:
    - image: 输入图像。

    返回:
    - 锐化后的图像。
    """
    #这是一个常见的锐化卷积核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    """
       该函数使用OpenCV库对图像进行二维卷积滤波。具体功能如下：
       cv2.filter2D：对图像应用卷积核。
       image：输入图像。
       -1：表示输出图像与输入图像具有相同的深度。
       kernel：用于滤波的卷积核。
    """
    return cv2.filter2D(image, -1, kernel)


# 创建保存结果的目录
result_dir = 'A3_Result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读取图像
image = cv2.imread('A3/001.3-bin.png')

# 应用不同的滤波并保存结果
methods = ['mean', 'gaussian', 'median', 'bilateral']
for method in methods:
    if method == 'median':
        # 中值模糊只需要一个整数作为核大小
        filtered_image = apply_blur(image, method=method, kernel_size=5)
    else:
        # 其他模糊方法使用元组
        filtered_image = apply_blur(image, method=method, kernel_size=(5, 5), sigma=1, diameter=9, sigma_color=75,
                                    sigma_space=75)

    # 锐化模糊后的图像
    sharpened_image = apply_sharpen(filtered_image)

    # 保存模糊后的图像
    blur_file_path = os.path.join(result_dir, f'{method}_filtered.jpg')
    cv2.imwrite(blur_file_path, filtered_image)

    # 保存锐化后的图像
    sharpen_file_path = os.path.join(result_dir, f'{method}_sharpened.jpg')
    cv2.imwrite(sharpen_file_path, sharpened_image)

    # 显示结果
    cv2.imshow(f'{method} filtered_image', filtered_image)
    cv2.imshow(f'{method} sharpened_image', sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
