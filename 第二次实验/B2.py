#A2.1的图片用于行切分，切分完放在A2.2下
# dataset下的图片用于自适应二值化放在B2下，注意观察实验结果

import cv2
import os

# 输入图像目录
input_dir = 'dataset\\dataset\\train\\image'
# 输出图像目录
output_dir = 'B2'

# 确保输出目录存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取输入目录中的所有文件名
image_files = os.listdir(input_dir)
print(image_files)
print(len(image_files))
print(image_files[1])

# 遍历所有图像文件
for image_file in image_files:
    # 构建完整的输入文件路径
    input_path = os.path.join(input_dir, image_file)
    print(input_path)
    # 读取图像
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)#以灰度模式读取图像
    print(img_gray)
    if img_gray is not None:
        # 自适应二值化处理（高斯加权平均值法）
        binary_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, blockSize=11, C=2)

        # 构建完整的输出文件路径
        output_path = os.path.join(output_dir, image_file)
        print(f"output_file:{output_path}")
        # 保存二值化图像
        cv2.imwrite(output_path, binary_img)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Failed to read: {input_path}")

print("Processing complete.")
