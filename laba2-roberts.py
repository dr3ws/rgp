import cv2 as opencv
import numpy as np

# img_before = opencv.imread('img/2.png', opencv.IMREAD_GRAYSCALE)
img_before = opencv.imread('img/image.png', opencv.IMREAD_GRAYSCALE)
i = img_before

rows = np.size(i, 0)
columns = np.size(i, 0)
img_after = np.zeros(i.shape)

for r in range(0, rows - 2):
    for c in range(0, columns - 2):
        H1 = int(i[r, c]) - i[r + 1, c + 1]
        H2 = int(i[r + 1, c]) - i[r, c + 1]
        img_after[r + 1, c + 1] = np.sqrt(np.square(H1) + np.square(H2))

img_after = img_after / np.max(img_after) * 255
img_after = img_after.astype('uint8')

img_result = opencv.hconcat([i, img_after])
opencv.imshow('Result Roberts', img_result)
opencv.waitKey(0)
