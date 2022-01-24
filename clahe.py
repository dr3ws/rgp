import cv2 as opencv
import numpy as np
from matplotlib import pyplot as plt

num = 0


def draw_graph(arr, check, last=False):  # отрисовка гистограммы
    global num

    plt.bar(np.arange(len(arr)), arr)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")

    if check == 1:
        plt.title('Histogram for 1 block:')
        plt.savefig('output/hists/histblock/histblock_' + str(num) + '.jpg', pad_inches=0)
        plt.show()
    if check == 2:
        plt.title('Histogram with limit:')
        plt.savefig('output/hists/histlimit/histlimit_' + str(num) + '.jpg', pad_inches=0)
        plt.show()
    if check == 3:
        plt.title('Normalized histogram:')
        plt.savefig('output/hists/histnorm/histnorm_' + str(num) + '.jpg', pad_inches=0)
        plt.show()
    if check == 4:
        plt.title('Cumulative histogram:')
        plt.savefig('output/hists/histcumul/histcumul_' + str(num) + '.jpg', pad_inches=0)
        plt.show()

    if last:
        num += 1

    # plt.savefig('output/hists/result.jpg', pad_inches=0)
    # plt.show()


def draw_image(img, index):  # отрисовка изображения
    plt.tick_params(labelsize=0, length=0)
    plt.imshow(img, cmap='gray')
    plt.savefig('output/source_' + str(index) + '.jpg', pad_inches=0)
    plt.show()


def handle_tile(img, top, bottom, left, right, clip_limit, exs):  # обработка блока
    hist = create_hist(img, top, bottom, left, right, clip_limit, exs)
    norm_hist = create_norm_hist(hist, count_total_number_of_pixels(top, bottom, left, right))
    cum_hist = create_cum_hist(norm_hist)
    modified_img = modify_image(img, top, bottom, left, right, cum_hist)
    return modified_img


def count_total_number_of_pixels(top, bottom, left, right):  # общее количество пикселей
    return (bottom - top + 1) * (right - left + 1)


def create_hist(img, top, bottom, left, right, clip_limit, exs):  # создание гистограммы для блока
    result = [0] * 256
    for i in range(top, bottom):
        for j in range(left, right):
            result[img[i, j]] = result[img[i, j]] + 1

    # draw_graph(result, 1, False)
    return fix_hist(result, clip_limit, exs)


def fix_hist(hist, clip_limit, exs):  # обработка гистограммы с условием лимита
    result = hist
    num_of_exs_pix = 0

    for i in range(0, 255):
        if result[i] > clip_limit:
            num_of_exs_pix += result[i] - clip_limit
            result[i] = clip_limit

    i = 0
    while num_of_exs_pix > 0:
        if result[i] < clip_limit:
            result[i] += 1
            num_of_exs_pix -= 1
        i = (i + 1) if i < 256 else 0

    while exs > 0:
        for i in range(0, 255):
            if result[i] < clip_limit:
                result[i] += 1
        exs -= 1

    # draw_graph(result, 2, False)
    return result


def create_norm_hist(hist, total_num_of_pixels):  # создание нормализованной гистограммы
    result = hist
    for i in range(0, len(hist)):
        result[i] = hist[i] / total_num_of_pixels

    # draw_graph(result, 3, False)
    return result


def create_cum_hist(hist):  # создание кумулятивной гистограммы
    result = hist
    for i in range(1, len(hist)):
        result[i] = hist[i - 1] + hist[i]

    # draw_graph(result, 4, True)
    return result


def modify_image(img, top, bottom, left, right, cum_hist):  # модификация яркости пикселей ячеек
    result = np.zeros(img.shape)

    for i in range(top, bottom):
        for j in range(left, right):
            result[i, j] = cum_hist[img[i, j]] * 255 \
                if abs((cum_hist[img[i, j]] * 255) - img[i, j]) < 40 \
                else img[i, j] + 40 \
                if cum_hist[img[i, j]] * 255 >= img[i, j] \
                else img[i, j] - 40

    return result


def interpolate_img(img):  # интерполяция
    result = img.copy()
    image_height, image_width = img.shape
    number_of_pixels_blocks = 0

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            m1 = np.array([[j, i - 1, (i - 1) * j, 1],
                           [j + 1, i, i * (j + 1), 1],
                           [j, i + 1, (i + 1) * j, 1],
                           [j - 1, i, i * (j - 1), 1]])
            v1 = np.array([img[i - 1, j],
                           img[i, j + 1],
                           img[i + 1, j],
                           img[i, j - 1]])
            arr = np.linalg.lstsq(m1, v1)
            result[i, j] = int(arr[0][0] * j + arr[0][1] * i + arr[0][2] * i * j + arr[0][3])
        number_of_pixels_blocks += image_width

    return result


def main_func():
    exs = 0
    block_size = 16    # размер блока (2, 4, 8 и 16)
    clip_limit = 32     # лимит максимального количества пикселей (1, 16, 32, 128 и 255)

    image = opencv.imread('img/source.jpg', opencv.IMREAD_GRAYSCALE)
    draw_image(image, 1)

    image_height, image_width = image.shape
    top = 0
    bottom = block_size - 1

    modified_img = np.zeros(image.shape)
    while bottom < image_height:
        left = 0
        right = block_size - 1
        while right < image_width:
            modified_img += handle_tile(image, top, bottom, left, right, clip_limit, exs)
            left = right
            right = right + block_size if (right + block_size < image_width) or (right == image_width - 1) \
                else image_width - 1
        top = bottom
        bottom = bottom + block_size if (bottom + block_size < image_height) or (bottom == image_height - 1) \
            else image_height - 1

    draw_image(modified_img, 2)

    interpolated_image = interpolate_img(modified_img)
    draw_image(interpolated_image, 3)


if __name__ == '__main__':
    np.warnings.filterwarnings('ignore')
    main_func()
