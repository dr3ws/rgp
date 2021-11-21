import cv2 as opencv
import numpy as np

# img_before = opencv.imread('img/2.png', opencv.IMREAD_GRAYSCALE)
# img_before = opencv.imread('img/2-1.png', opencv.IMREAD_GRAYSCALE)
img_before = opencv.imread('img/image.png', opencv.IMREAD_GRAYSCALE)

x = np.array([[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]])

y = np.array([[-1, -1, -1],
              [0, 0, 0],
              [1, 1, 1]])

rows = np.size(img_before, 0)
columns = np.size(img_before, 1)
img_after = np.zeros(img_before.shape)

for r in range(0, rows - 2):
    for c in range(0, columns - 2):
        h1 = sum(sum(x * img_before[r:r + 3, c:c + 3]))
        h2 = sum(sum(y * img_before[r:r + 3, c:c + 3]))
        img_after[r + 1, c + 1] = np.sqrt(np.square(h1) + np.square(h2))

img_after = img_after / np.max(img_after) * 255
img_after = img_after.astype('uint8')

img_result = opencv.hconcat([img_before, img_after])
opencv.imshow("Result Prewitt", img_result)
opencv.waitKey(0)
