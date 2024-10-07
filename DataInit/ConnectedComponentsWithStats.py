import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
src = cv2.imread("D:\\Users\\xiaokunhao\\PycharmProjects\\Image_Processing\\DataInit\\granule.png")

# 高斯模糊
src = cv2.GaussianBlur(src, (3, 3), 0)

# 转换为灰度图像
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 阈值处理
ret, binary_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 使用开运算去掉外部的噪声
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.morphologyEx(binary_, cv2.MORPH_OPEN, kernel)

# 显示经过开运算处理后的二值图像
cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 连通组件分析
num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
print("Number of components:", num_labels)
print("Stats of each component:", stats)
print("Stats of each component shape:", stats.shape)
print("每个组件的中心位置:", centers)

# 生成随机颜色
colors = []
for i in range(num_labels):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    colors.append((b, g, r))
print(colors)

# 设置背景颜色为黑色
colors[0] = [0, 0, 0]

# 复制源图像
image = np.copy(src)

# 标记连通组件
for t in range(1, num_labels):
    x, y, w, h, area = stats[t]
    cx, cy = centers[t]
    # 标出每个组件的中心位置
    cv2.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
    # 画出外接矩形
    cv2.rectangle(image, (x, y), (x + w, y + h), colors[t], 1, 8, 0)
    # 标记组件编号
    cv2.putText(image, "No." + str(t), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255))
    print("label index %d, area of the label : %d" % (t, area))

# 显示并保存标记后的图像
cv2.imshow("colored labels", image)
cv2.imwrite("labels.png", image)
print("total number:", num_labels - 1)

cv2.waitKey(0)
cv2.destroyAllWindows()
