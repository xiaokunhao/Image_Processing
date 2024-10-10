import numpy as np
import matplotlib.pyplot as plt
import cv2

def noise_type(img):
    """
    判断图像的噪声类型

    Args:
        img: 图像

    Returns:
        噪声类型
    """

    # 计算图像的均值和方差
    img_mean = np.mean(img)
    img_var = np.var(img)

    print("图像均值为：",img_mean)
    print("图像方差为：",img_var)

    # 计算图像的峰度和偏度
    img_kurtosis = np.mean((img - img_mean)**4) / (img_var**2)
    img_skewness = np.mean((img - img_mean)**3) / (img_var**1.5)
    print("图像的峰度为",img_kurtosis)
    print("图像的偏度为", img_skewness)

    # 判断噪声类型
    if img_mean == 0 and img_var == img_var.max():
        return "高斯白噪声"
    elif img_mean != 0 and img_var == img_var.max():
        return "均值偏移高斯白噪声"
    elif img_kurtosis == 3 and img_skewness == 0:
        return "高斯噪声"
    elif img_kurtosis == 3.69 and img_skewness == 1.58:
        return "拉普拉斯噪声"
    elif img_kurtosis == 0 and img_skewness == 0:
        return "指数噪声"
    else:
        return "其他噪声"
def plot_hist(img):
    """
    绘制图像直方图

    Args:
        img: 图像
    """

    plt.hist(img, bins=256)
    plt.show()

if __name__ == "__main__":
    # 读取图像
    img = cv2.imread("Images/input-resize_480360/1.bmp",0)

    # 判断噪声类型
    noise_type = noise_type(img)
    print("噪声类型：", noise_type)

    # 绘制直方图
    plot_hist(img)
