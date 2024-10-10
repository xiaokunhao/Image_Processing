#A2.1的图片用于行切分，切分完放在A2.2下
# dataset下的图片用于自适应二值化放在B2下，注意观察实验结果
import cv2
import numpy as np
import os

def split_lines(input_path, output_dir):
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 读取图片并转换为灰度图像
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用二值化方法进行图像处理
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # 计算水平投影
    horizontal_projection = np.sum(binary, axis=1)

    # 使用水平投影来检测文本行
    line_indices = []
    in_line = False
    for i, value in enumerate(horizontal_projection):
        if value > 0 and not in_line:
            in_line = True
            start_index = i
        elif value == 0 and in_line:
            in_line = False
            end_index = i
            line_indices.append((start_index, end_index))

    # 切分每一行并保存
    for idx, (start, end) in enumerate(line_indices):
        line_img = img[start:end, :]
        slice_path = os.path.join(output_dir, f'line_{idx + 1}.png')
        cv2.imwrite(slice_path, line_img)

# 使用示例
input_image_path = 'A2.1\\tahiti.png'  # 将这里替换为你的图片路径
output_directory = 'A2.2'

split_lines(input_image_path, output_directory)
