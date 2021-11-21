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

a = 0
b = hist1.size - 1
c = 0
d = 255
# нижняя граница исходного диапозона
while hist1[a] == 0:
    a += 1
# верхняя граница исходного диапозона
while hist1[b] == 0:
    b -= 1

i_in = 0
table = []
for i_in in range(0, 256):
    i_out = (i_in - a) * ((d - c) / (b - a)) + c
    table.append(round(i_out))
    i_in += 1

img_after = img_before.copy()
for i in range(img_after.shape[0]):
    for j in range(img_after.shape[1]):
        img_after[i, j] = table[img_before[i, j]]

hist2, bins = np.histogram(img_after.ravel(), 256, [0, 255])

plt.plot(hist2, 'r')
plt.title('Histogram:')
plt.show()

img_result = opencv.hconcat([img_before, img_after])
opencv.imshow("Result", img_result)
opencv.waitKey(0)
