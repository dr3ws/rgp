import cv2 as opencv
import numpy as np


def dilation(a_, b_):  # дилатация
    B_h, B_w = b_.shape
    img = np.copy(a_)
    target_row, target_col = np.where(a_ == 255)
    # для каждого пикселя ищем пересечение
    for row, col in zip(target_row, target_col):
        for i in range(B_h):
            for j in range(B_w):
                checking_row = row + i - B_h // 2
                checking_col = col + j - B_w // 2
                if checking_row >= 0 and checking_col >= 0:
                    try:
                        img[checking_row][checking_col] = b_[i][j]  # якорная точка
                    except IndexError:
                        pass
    return img


def erosion(a_, b_):  # эрозия
    B_h, B_w = b_.shape
    img = np.copy(a_)
    target_row, target_col = np.where(a_ == 255)
    # ищем все смещения, при котором B полностью входит в A
    for row, col in zip(target_row, target_col):
        for i in range(B_h):
            for j in range(B_w):
                checking_row = row + i - B_h // 2
                checking_col = col + j - B_w // 2
                if checking_row >= 0 and checking_col >= 0:
                    try:
                        # если B не является подмножеством A, то точка равна 0
                        if a_[checking_row][checking_col] != b_[i][j]:
                            img[row][col] = 0
                    except IndexError:
                        img[row][col] = 0
                else:
                    img[row][col] = 0
    return img


def opening(a_, b_):  # открытие
    temp_erosion = erosion(a_, b_)
    img = dilation(temp_erosion, b_)
    return img


def closing(a_, b_):  # закрытие
    temp_dilation = dilation(a_, b_)
    img = erosion(temp_dilation, b_)
    return img


img_before = opencv.imread('img/4.jpg', 0)
ret, binary = opencv.threshold(img_before, 128, 255, opencv.THRESH_BINARY)  # бинаризация
opencv.imwrite('img/4/binary.jpg', binary)

a = binary
b = np.array(
    [[255, 255, 255],
     [255, 255, 255],
     [255, 255, 255]], dtype=np.uint8)

img_dilation = dilation(a, b)
opencv.imwrite('img/4/dilation.jpg', img_dilation)

img_erosion = erosion(a, b)
opencv.imwrite('img/4/erosion.jpg', img_erosion)

img_opening = opening(a, b)
opencv.imwrite('img/4/opening.jpg', img_opening)

img_closing = closing(a, b)
opencv.imwrite('img/4/closing.jpg', img_closing)

opencv.waitKey(0)
