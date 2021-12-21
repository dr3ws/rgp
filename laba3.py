import cv2 as opencv
import numpy as np
import math


def gaussian_noise(image, sigm):
    n = np.random.normal(0, sigm, image.shape)
    n = image + n
    return n


def calc_diff(img1, img2):
    rows, cols = img1.shape
    square = rows * cols
    return np.sum(np.abs(img1.astype(np.float64) - img2.astype(np.float64))) / square

# ================================================== Gaussian Blur =================================================== #


def gaussian_filter(image, kernel_size, sigm):  # kernel_size=3, sigm=1.3
    H, W = image.shape

    # Zero padding
    pad = kernel_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=float)
    out[pad: pad + H, pad: pad + W] = image.copy().astype(float)

    # prepare Kernel
    K = np.zeros((kernel_size, kernel_size), dtype=float)
    for x in range(-pad, -pad + kernel_size):
        for y in range(-pad, -pad + kernel_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigm ** 2)))

    K /= (2 * np.pi * (sigm ** 2))
    K /= K.sum()
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + kernel_size, x: x + kernel_size])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out

# ================================================== Bilateral Filter ================================================ #


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def dnorm(x, sigm):
    return (1.0 / (2 * math.pi * (sigm ** 2))) * math.exp(-(x ** 2) / (2 * sigm ** 2))


def calc_pixel(image, x, y, filter_size, sigm_r, sigm_s):
    fsum = 0
    Wp = 0
    for i in range(filter_size):
        for j in range(filter_size):
            neighbour_x = int(x - (filter_size / 2 - i))
            neighbour_y = int(y - (filter_size / 2 - j))
            gr = dnorm(image[neighbour_x, neighbour_y] - image[x, y], sigm_r)
            gs = dnorm(distance(neighbour_x, neighbour_y, x, y), sigm_s)

            weight = gr * gs
            fsum += image[neighbour_x, neighbour_y] * weight
            Wp += weight
    return int(round(fsum / Wp))


def bilateral_filter(image, filter_size, sigm_r, sigm_s):
    output = np.zeros(image.shape)
    rows, cols = image.shape
    pad = filter_size // 2
    padded_img = np.zeros((rows + (2 * pad), cols + (2 * pad)))
    padded_img[pad:padded_img.shape[0] - pad, pad:padded_img.shape[1] - pad] = image
    for i in range(rows):
        for j in range(cols):
            output[i, j] = calc_pixel(padded_img, i + pad, j + pad, filter_size, sigm_r, sigm_s)
            if output[i, j] > 255:
                output[i, j] = 255
            if output[i, j] < 0:
                output[i, j] = 0
    return output

# =================================================== NL-Means Filter ================================================ #


def nl_means_filter(image, sigm, patch_big, patch_small):  # 7 3
    # разбиваем на большое и маленькое окно для обхода вокруг центра пикселя
    # создаем пэддинг для изображения
    pad = patch_big + patch_small

    img_1 = np.pad(image, pad, mode='reflect')

    result_img = np.zeros((image.shape[0], image.shape[1]))
    H, W = img_1.shape

    # коэффициент. для распределения Гаусса (параметр разрброса весов)
    # задается вручную
    H_gauss = 25

    # проход по всему изображению
    for y in range(pad, H - pad):
        for x in range(pad, W - pad):

            current_val = 0

            # получаем большое окно для сдвига по картинке
            startY = y - patch_big
            endY = y + patch_big

            startX = x - patch_big
            endX = x + patch_big

            # нормирующий делитель и максимальное значение веса
            Wp, maxweight = 0, 0
            # обойти по всем соседям пикселя окна и найти похожее значение, рассчитать вес
            # аккумулировать вес и добавить в текущую картинку
            for ypix in range(startY, endY):
                for xpix in range(startX, endX):
                    # создаем для текущего пикселя окна для рассчета gauss - L2 norm
                    window1 = img_1[y - patch_small:y + patch_small, x - patch_small:x + patch_small].copy()
                    window2 = img_1[ypix - patch_small:ypix + patch_small, xpix - patch_small:xpix + patch_small].copy()

                    # используем экспоненциальное ядро для вычисления весов для текущего пикселя
                    weight = np.exp(-(np.sum((window1 - window2) ** 2) + 2 * (sigm ** 2)) / (H_gauss ** 2))

                    # находим максимальное значение веса
                    if weight > maxweight:
                        maxweight = weight

                    # если текущий пиксель совпадает с нужным нам похожим пикселем вес будет равен максимальному
                    if (y == ypix) and (x == xpix):
                        weight = maxweight

                    Wp += weight
                    current_val += weight * img_1[ypix, xpix]
            # обновляем изображения с учетом нового веса
            result_img[y - pad, x - pad] = current_val / Wp
    return result_img


def writeres(noisy):
    print('\nGaussian:')
    for sigma in range(1, 6, 1):
        GF = gaussian_filter(noisy, 6, sigma)
        opencv.imwrite('img/3/gaussian/s=' + str(sigma) + '.jpg', GF)
        print('σ =', sigma, '\tMAE =', calc_diff(noisy, GF))

    print('\nBilateral:')
    for sigma_spatial in range(4, 16, 4):
        print('σ_s =', sigma_spatial)
        for sigma_r in range(16, 96, 16):
            BF = bilateral_filter(noisy, 6, sigma_r, sigma_spatial)
            opencv.imwrite('img/3/bilateral/s_s=' + str(sigma_spatial) + '/s_r=' + str(sigma_r) + '.jpg', BF)
            print('  σ_r =', sigma_r, '\tMAE =', calc_diff(noisy, BF))

    print('\nNL-Means:')
    print('Local Window: 1, window: 15')
    for sigma in range(5, 50, 15):
        NLMF = nl_means_filter(noisy, sigma, 15, 1)
        opencv.imwrite('img/3/nlm/1_15/s=' + str(sigma) + '.jpg', NLMF)
        print('σ =', sigma, '\tMAE =', calc_diff(noisy, NLMF))

    print('Local Window: 2, window: 5')
    for sigma in range(5, 50, 15):
        NLMF = nl_means_filter(noisy, sigma, 5, 2)
        opencv.imwrite('img/3/nlm/2_5/s=' + str(sigma) + '.jpg', NLMF)
        print('σ =', sigma, '\tMAE =', calc_diff(noisy, NLMF))

    print('Local Window: 3, window: 10')
    for sigma in range(5, 50, 15):
        NLMF = nl_means_filter(noisy, sigma, 10, 3)
        opencv.imwrite('img/3/nlm/3_10/s=' + str(sigma) + '.jpg', NLMF)
        print('σ =', sigma, '\tMAE =', calc_diff(noisy, NLMF))


def comparison(image):
    print('\nGaussian:')
    for sigma in range(3, 15, 3):
        res = opencv.imread('img/3/gaussian/s=' + str(sigma) + '.jpg', opencv.IMREAD_GRAYSCALE)
        print('σ =', sigma, '\tI_before and I_res MAE =', calc_diff(image, res))

    print('\nBilateral:')
    for sigma_spatial in range(4, 16, 4):
        print('σ_s =', sigma_spatial)
        for sigma_r in range(16, 96, 16):
            res = opencv.imread('img/3/bilateral/s_s=' + str(sigma_spatial) + '/s_r=' + str(sigma_r) + '.jpg',
                                opencv.IMREAD_GRAYSCALE)
            print('  σ_r =', sigma_r, '\tI_before and I_res MAE =', calc_diff(image, res))

    print('\nNL-Means:')
    print('Local Window: 1, window: 15')
    for sigma in range(5, 50, 15):
        res = opencv.imread('img/3/nlm/1_15/s=' + str(sigma) + '.jpg', opencv.IMREAD_GRAYSCALE)
        print('σ =', sigma, '\tI_before and I_res MAE =', calc_diff(image, res))

    print('Local Window: 2, window: 5')
    for sigma in range(5, 50, 15):
        res = opencv.imread('img/3/nlm/2_5/s=' + str(sigma) + '.jpg', opencv.IMREAD_GRAYSCALE)
        print('σ =', sigma, '\tI_before and I_res MAE =', calc_diff(image, res))

    print('Local Window: 3, window: 10')
    for sigma in range(5, 50, 15):
        res = opencv.imread('img/3/nlm/3_10/s=' + str(sigma) + '.jpg', opencv.IMREAD_GRAYSCALE)
        print('σ =', sigma, '\tI_before and I_res MAE =', calc_diff(image, res))


img_before = opencv.imread('img/3.jpg', opencv.IMREAD_GRAYSCALE)
noise = gaussian_noise(img_before, 5)
opencv.imwrite('img/3/img_source.jpg', img_before)
opencv.imwrite('img/3/img_noise.jpg', noise)

# writeres(noise)
# comparison(img_before)
