import cv2 as opencv
import numpy as np
from matplotlib import pyplot as plt

# считываем картинку в режиме оттенков серого
img_before = opencv.imread('img/1.jpg', opencv.IMREAD_GRAYSCALE)
# создаём гистограмму данных изображения
plt.hist(img_before.ravel(), 256, [0, 255])
# hist - массив, где значение соответствует количеству пикселей в изображении с соответствующим значением пикселя
# bins - диапозон по оси Х
hist1, bins = np.histogram(img_before.ravel(), 256, [0, 255])

MN = len(img_before.ravel())  # количество пикселей
L = len(hist1)  # количество уровней серого
cumulative = hist1.cumsum()

cdf = ((cumulative - cumulative[0]) / (MN - 1) * (L - 1))
cdf = cdf.astype(np.uint8)

img_after = cdf[img_before]
hist2, bins2 = np.histogram(img_after, 256, [0, 255])
cdf2 = hist2.cumsum()
plt.plot(hist2, 'g')

img_result = opencv.hconcat([img_before, img_after])
opencv.imshow("Result", img_result)
plt.show()
opencv.waitKey(0)
