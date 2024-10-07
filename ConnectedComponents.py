import cv2
import numpy as np
import matplotlib.pyplot as plt
#connectedComponentsWithStats()：输入图像，必须是二值图，即8位单通道图像。（因此输入图像必须先进行二值化处理才能被这个函数接受
# 读取图像
src=cv2.imread('D:\\Users\\xiaokunhao\\PycharmProjects\\Image_Processing\\DataInit\\Pill1.jpg')
src=cv2.GaussianBlur(src,(3,3),0)
gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)#函数将输入图像 src 从 BGR（蓝绿红）色彩空间转换为灰度图 gray。主要功能如下：读取输入图像 src。指定参数 cv2.COLOR_BGR2GRAY 表示从 BGR 转换为灰度图
ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("binary",binary)
cv2.imwrite('binary.png',binary)#把二值化图保存在本地
#该函数使用OpenCV库计算二值图像binary的连通组件。功能包括：返回值output包含连通组件的数量及标签图像。
output=cv2.connectedComponents(binary,connectivity=8,ltype=cv2.CV_32S)
num_labels=output[0]  #组件数量=num_labels-1
labels=output[1]
print(f'labels.shape：{labels.shape}')
# 构造颜色
#这里生成了一组随机颜色，用于标记不同的连通组件。背景颜色被设置为黑色 (0, 0, 0)。
colors = []
for i in range(num_labels):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    colors.append((b, g, r))
colors[0] = (0, 0, 0)


# 画出连通图,绘制连通区域标记图
h, w = gray.shape
print("二值图尺寸：",gray.shape)
#创建一个空的彩色图像 (image)，并将每个像素点根据其标签赋予相应的颜色
image=np.zeros((h,w,3),dtype=np.uint8)
for row in range(h):
    for col in range(w):
        image[row,col]=colors[labels[row,col]]
        print(colors[labels[row,col]])
print(colors)
cv2.imshow("color images",image)
cv2.imwrite("labels.png",image)
print("total componets:",num_labels-1)
cv2.waitKey(0)
cv2.destroyAllWindows()














"""
image = cv2.imread('D:\\Users\\xiaokunhao\\PycharmProjects\\Image_Processing\\DataInit\\liu.jpg', 0) # 0表示加载为灰度图像
# 读取图像
# 对图像进行二值化处理
#调用 cv2.threshold() 函数返回两个参数：
#阈值_：计算出的实际阈值（如果你使用的是像 cv2.THRESH_OTSU 这样的自动阈值方法）。
#处理后的图像：经过二值化处理后的图像。
#在给定的代码片段中，_ 通常用来忽略第一个返回值，即实际计算出的阈值；thresholded 是经过二值化处理后的图像。
_, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(_)
# 使用 matplotlib 显示图像
plt.imshow(thresholded, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')
plt.show()
# 进行连通组件分析
#“连通组件”指的是图像中相邻且具有相同属性（如相同的像素值）的像素集合。
# 在图像处理中，通过分析这些连通组件，可以识别和分割出图像中的不同对象或区域。
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded)
# 输出结果
print("Number of components:", num_labels)
print("Stats of each component:", stats)
print("Stats of each component shape:",stats.shape)
print("Centroids of each component:", centroids)

"""


