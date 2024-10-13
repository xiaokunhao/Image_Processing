import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 创建结果目录
def create_result_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# 读取灰度图像
def read_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError("Image not found")
        return image
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def adaptive_threshold(image, block_size, C):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, C)

def save_image(image, block_size, C, result_dir):
    filename = os.path.join(result_dir, f'image_bs{block_size}_C{C}.png')
    cv2.imwrite(filename, image)

def evaluate_image_quality(binarized_image):
    """
    评估二值化图像的质量。简单的评估方法是计算图像的对比度。
    具体可以根据需求定义更复杂的评估标准。
    """
    # 计算前景和背景的平均值
    foreground = binarized_image[binarized_image == 255]
    background = binarized_image[binarized_image == 0]

    if len(foreground) == 0 or len(background) == 0:
        return 0  # 如果只有前景或背景，质量较差

    foreground_mean = np.mean(foreground)
    background_mean = np.mean(background)

    # 计算对比度
    contrast = abs(foreground_mean - background_mean)
    return contrast

# 参数设置
block_sizes = [11, 15, 21]
Cs = [2, 5, 10]

# 创建结果目录
result_dir = create_result_directory('Results')

# 读取灰度图像
image_path = 'dataset/dataset/train/image/image_0.jpg'
image = read_image(image_path)
if image is None:
    exit(1)

# 创建一个字典来存储评估结果
quality_results = {}

# 对不同参数组合进行二值化处理并保存结果
for block_size in block_sizes:
    for C in Cs:
        binarized_image = adaptive_threshold(image, block_size, C)
        save_image(binarized_image, block_size, C, result_dir)

        # 评估图像质量
        quality = evaluate_image_quality(binarized_image)
        quality_results[(block_size, C)] = quality

# 输出评估结果
for params, quality in sorted(quality_results.items(), key=lambda x: x[1], reverse=True):
    print(f'blockSize={params[0]}, C={params[1]}, Quality={quality}')

# 显示对比结果
plt.figure(figsize=(10, 10))
for i, (block_size, C) in enumerate(quality_results.keys()):
    binarized_image_path = os.path.join(result_dir, f'image_bs{block_size}_C{C}.png')
    binarized_image = cv2.imread(binarized_image_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(len(block_sizes), len(Cs), i + 1)
    plt.imshow(binarized_image, cmap='gray')
    plt.title(f'bs={block_size}, C={C}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 最佳参数组合
best_params = max(quality_results, key=quality_results.get)
print(f'Best parameters: blockSize={best_params[0]}, C={best_params[1]}')
