import os
import numpy as np
import cv2


def get_vvList(list_data):
    # 取出list中像素存在的区间
    vv_list = list()  # 调用 list() 函数创建一个空列表
    v_list = list()
    for index, i in enumerate(list_data):
        if i > 0:
            # 如果当前元素大于0（即存在黑色像素），将其索引添加到 v_list
            v_list.append(index)
        else:
            # 如果当前元素不大于0（即不存在黑色像素)
            if v_list:
                vv_list.append(v_list)
                # 检查 v_list 是否为空。如果非空，意味着前面有一段连续的非零元素（黑色像素）。
                # 将 v_list 添加到 vv_list 中，以保存这段连续非零元素的索引区间。
                # 使用 v_list = [] 创建一个新的空列表，以准备记录下一段连续非零元素的索引。
                v_list = []
    if v_list:  # 确保最后一个非零区间被添加到 vv_list 中
        vv_list.append(v_list)
    print("vv_list:", vv_list)
    print(len(vv_list))
    return vv_list


if __name__ == '__main__':
    input_dir = 'A2.1'
    output_dir = 'A2.2'
    input_file = os.path.join(input_dir, 'tahiti.png')

    # 创建输出目录如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_bgr = cv2.imread(input_file, 1)
    if img_bgr is not None:
        img = img_bgr.copy()  # 保存图像副本
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值化
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        # 显示二值化图像
        cv2.imshow("Binary", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        rows, cols = binary.shape
        print("二值化图像：", binary.shape)

        hor_list = [0] * rows  # 行列表
        for i in range(rows):
            for j in range(cols):
                # 统计每一行的黑色像素总数
                if binary.item(i, j) == 0:
                    hor_list[i] = hor_list[i] + 1

        print(f'hor_list: {hor_list}')
        print(f'hor_list长度: {len(hor_list)}')

        hor_arr = np.array(hor_list)
        hor_arr[np.where(hor_arr < 5)] = 0  # 将黑色像素少于 5 的行视为噪声并置为零
        hor_list = hor_arr.tolist()

        # 绘制水平投影
        img_white = np.ones(shape=(rows, cols), dtype=np.uint8) * 255

        for i in range(rows):
            pt1 = (cols - 1, i)
            pt2 = (cols - 1 - hor_list[i], i)
            print(f'pt1: {pt1}, pt2: {pt2}')
            cv2.line(img_white, pt1, pt2, (0,), 1)

        cv2.imshow('水平投影', img_white)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 取出各个文字区间
        vv_list = get_vvList(hor_list)
        print("hor_list列表长度：", len(hor_list))
        print(vv_list)

        # 保存切分后的结果到 A2.2 目录
        for idx, i in enumerate(vv_list):
            img_hor = img_bgr[i[0]:i[-1], :, :]
            output_file = os.path.join(output_dir, f'line_{idx + 1}.png')

            cv2.imwrite(output_file, img_hor)

            # 显示切分结果（可选）
            cv2.imshow(f'line{idx + 1}', img_hor)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
