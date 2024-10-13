import cv2
import numpy as np
import os

def overlap_tile(img, h, w, stride_h, stride_w):
    """
    将图像分割成有重叠的小块，并记录切块的位置信息。

    参数:
    img: 待切割的图像（ndarray）
    h: 切块的高度
    w: 切块的宽度
    stride_h: y方向的步长
    stride_w: x方向的步长

    返回:
    tiles: 包含所有切块的列表
    tile_positions: 包含所有切块位置的列表
    """
    tiles = []
    tile_positions = []
    height, width = img.shape[:2]

    for y in range(0, height - h + stride_h, stride_h):
        for x in range(0, width - w + stride_w, stride_w):
            y_max = min(y + h, height)
            x_max = min(x + w, width)
            tile = img[y:y_max, x:x_max]
            tiles.append(tile)
            tile_positions.append((y, x, y_max - y, x_max - x))  # 记录位置和大小

    return tiles, tile_positions

def reconstruct_image(tiles, tile_positions, height, width):
    """
    根据切块和位置信息重建图像。

    参数:
    tiles: 包含所有切块的列表
    tile_positions: 包含所有切块位置的列表
    height: 原始图像的高度
    width: 原始图像的宽度

    返回:
    recon_img: 重建后的图像
    """
    recon_img = np.zeros((height, width, 3), dtype=np.uint8)

    for tile, (y, x, tile_h, tile_w) in zip(tiles, tile_positions):
        recon_img[y:y+tile_h, x:x+tile_w] = tile

    return recon_img

# 使用示例
if __name__ == "__main__":
    # 读取图像
    img_path = 'tahiti.png'
    img = cv2.imread(img_path)

    # 设置切块参数
    h, w = 256, 256  # 切块的高度和宽度
    stride_h, stride_w = 128, 128  # 步长

    # 获取切块和位置信息
    tiles, tile_positions = overlap_tile(img, h, w, stride_h, stride_w)

    # 展示切块结果（可选）
    for i, tile in enumerate(tiles):
        cv2.imshow(f'Tile {i}', tile)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 重建图像
    recon_img = reconstruct_image(tiles, tile_positions, img.shape[0], img.shape[1])

    # 显示和保存重建后的图像
    cv2.imshow('Reconstructed Image', recon_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('reconstructed_image.png', recon_img)

    # 保存切块结果到R3目录下（需要先创建该目录）
    output_dir = 'R3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, tile in enumerate(tiles):
        output_tile_path = os.path.join(output_dir, f'tile_{i}.png')
        cv2.imwrite(output_tile_path, tile)