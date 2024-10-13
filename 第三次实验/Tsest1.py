import cv2
import numpy as np

# 创建两个示例图像
img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# 将两个图像的某些区域设置为0（黑色）
img1[30:70, 30:70] = 0
img2[50:90, 50:90] = 0
cv2.imwrite('A3/test1.png',img1)
cv2.imwrite('A3/test2.png',img2)
# 执行按位与操作
result = cv2.bitwise_and(img1, img2)
cv2.imwrite('A3/test.png',result)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()