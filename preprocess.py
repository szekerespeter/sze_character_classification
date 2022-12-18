import os
import cv2
from skimage.filters import median
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

files = os.listdir("test_set\\")
for file in files:
    img_resized = cv2.imread("test_set\\" + file, cv2.IMREAD_GRAYSCALE)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(img_resized, cv2.MORPH_DILATE, se)

    out_gray = cv2.divide(img_resized, bg, scale=255)
    img_resized = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    img_resized = cv2.bitwise_not(img_resized)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

    img_resized = cv2.fastNlMeansDenoising(img_resized)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    result = cv2.imwrite("test_set3\\" + file, img_resized)

