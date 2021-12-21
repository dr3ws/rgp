import cv2 as opencv
import numpy as np
from matplotlib import pyplot as plt

histogram = []
check = 0


def cdfunc(img):  # comulative distribution function
    global histogram
    global check

    histogram = np.bincount(img.ravel(), minlength=256)
    MN = len(img.ravel())  # количество пикселей
    L = len(histogram)  # количество уровней серого
    cumulative = histogram.cumsum()

    cdf = ((cumulative - cumulative[0]) / (MN - 1) * (L - 1))
    cdf = cdf.astype(np.uint8)

    check += 1

    if check == 1:
        plt.plot(cdf, 'red')
    if check == 2:
        plt.plot(cdf, 'blue')
    if check == 3:
        plt.plot(cdf, 'orange')
    return cdf


def matching_histogram(img__1, img__2):  # метод для приведения гистограммы исх. изображения к целевому
    H_i = cdfunc(img__1)
    H_t = cdfunc(img__2)

    new_hist = np.zeros_like(H_i)
    new_img = np.zeros_like(img__2)

    for H_i_index, H_i_color in enumerate(H_i):
        min_difference = 257
        index = 0
        for H_t_index, H_t_color in enumerate(H_t):
            if abs(int(H_i_color) - H_t_color) <= min_difference:
                min_difference = abs(int(H_i_color) - H_t_color)
                index = H_t_index
        new_hist[H_i_index] = index

    for x in range(img__1.shape[0]):
        for y in range(img__1.shape[1]):
            val = int(img__1[x, y])
            new_img[x, y] = new_hist[val]

    cdfunc(new_img)
    plt.title('red-исходное, blue-целевое, orange-result')
    plt.show()
    opencv.imshow('Matching histogram result', new_img)


img_1 = opencv.imread('img/1-1.png', opencv.IMREAD_GRAYSCALE)  # исходное изображение
img_2 = opencv.imread('img/1-2.png', opencv.IMREAD_GRAYSCALE)  # целевое изображение
hist1, bins1 = np.histogram(img_1.ravel(), 256, [0, 255])
hist2, bins2 = np.histogram(img_2.ravel(), 256, [0, 255])
plt.plot(hist1, 'r')
plt.plot(hist2, 'b')
plt.title('Histogram:')
plt.show()

opencv.imshow('img_1', img_1)
opencv.imshow('img_2', img_2)
matching_histogram(img_1, img_2)
opencv.waitKey(0)
