import cv2
import numpy as np

# 读取图像
image_path = "A3/001.3-bin.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 二值化处理
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 黑底白字转换（反转颜色）
binary_image = cv2.bitwise_not(binary_image)
cv2.imshow("黑底白字",binary_image)
cv2.waitKey(0)


# 显示二值化图像
cv2.imshow("Binary Image", binary_image)

# 模糊处理
blurred_image = cv2.blur(binary_image, (5, 5))

# 创建锐化内核
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# 应用锐化滤波器
sharpened_image = cv2.filter2D(blurred_image, -1, sharpen_kernel)

# 显示和保存结果图像
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Sharpened Image", sharpened_image)

cv2.imwrite("blurred_image.png", blurred_image)
cv2.imwrite("sharpened_image.png", sharpened_image)

# 等待按键以关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
