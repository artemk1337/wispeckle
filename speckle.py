from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *
from scipy.ndimage import rotate
from random import randint
import cv2  # or from PIL import Image


"""
if use PIL:
    cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA) =>
    => Image.fromarray(img).resize((512, 512), Image.ANTIALIAS)
"""


def draw(img, s, vmax):
    if vmax != 0:
        plt.imsave(s, cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA), vmax=vmax, cmap='gray')
    else:
        plt.imsave(s, cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA), cmap='gray')


def save_binary(a=0.0, b=0.0):
    plt.imsave('binary.png', cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA), vmin=a, vmax=b,
               cmap='gray')
    plt.imshow(cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA), vmin=a, vmax=b, cmap='gray')
    plt.show()


def plot(array):
    plt.plot(array)
    plt.show()


"""<===============Part_1===============>"""


data = fits.open('speckledata.fits')[2].data

img = np.mean(data, axis=0, dtype=np.float64)

draw(img, 'mean.png', 0.0)


"""<===============Part_2===============>"""


# man
"""
fft2(a, s=None, axes=(-2, -1)) — прямое двухмерное ПФ.
ifft2(a, s=None, axes=(-2, -1)) — обратное двухмерное ПФ.
rfft2(a, s=None, axes=(-2, -1)) — прямое двухмерное ДПФ.
irfft2(a, s=None, axes=(-2, -1)) — обратное двухмерное ДПФ.
fftshift(x, axes=None) — преобразует массив (с результатом ДПФ, от функций fft*) так,
чтобы нулевая частота была в центре.
ifftshift(x, axes=None) — делает обратную операцию.
np.sum(a, axis=0)
"""

img = ifft2(data, s=None, axes=(-2, -1))

img = fftshift(img)

img = np.abs(img ** 2)

fourier_data = img

img = np.sum(img, axis=0)

fourier_sum = img

draw(img, 'fourier.png', 100)


"""<===============Part_3===============>"""


x = np.zeros((100, 200, 200))
for i in range(100):
    x[i] = rotate(fourier_sum, randint(-180, 180), reshape=False)

# Образ рассеяния
img = np.mean(x, axis=0)

rotaver_mean = img

draw(img, 'rotaver.png', 100)


img = np.divide(fourier_sum, rotaver_mean)
print(img.max(), img.min())
# plot(img)


"""<===============Part_4===============>"""


img[np.abs(img) >= 13] = 0

# plot(img)

img = ifft2(img, s=None, axes=(-2, -1))

img = ifftshift(img)

img = np.abs(img)

# save_binary(0.05, 0.07)
