import matplotlib.pyplot as plt
import numpy as np
import cv2
import bm3d

img = cv2.imread('house.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray')
plt.show()

# calculate psnr
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((255 ** 2) / mse)


# add gaussian noise
img_noise = np.copy(img)
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        img_noise[i][j] += np.random.normal(scale = 20.0)


plt.imshow(img_noise, cmap='gray')
plt.show()

print(psnr(img, img_noise))

# part 2

def gaussian_weighting(x1, y1, x2, y2, sigma):
    mse = ((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2.0
    return np.exp(-0.5 * mse / (sigma ** 2))

denoised = np.zeros(img_noise.shape, dtype=float)

sigmas = [1, 5, 25]

for sigma in sigmas:
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            numerator = 0.0
            denominator = 0.0

            for w1 in range(-12, 13):
                for w2 in range(-12, 13):
                    if not(w1 + i < 0 or w1 + i >= img.shape[0] or w2 + j < 0 or w2 + j >= img.shape[1]):
                        weight = gaussian_weighting(i, j, i + w1, j + w2, sigma)
                        numerator += img_noise[i + w1][j + w2] * weight
                        denominator += weight

            #print(numerator, denominator)
            denoised[i][j] = numerator/denominator

    print(psnr(img, denoised))

    plt.imshow(denoised, cmap='gray')
    plt.show()

# part 5
sigmas = [5 / 255, 20 / 255, 50 / 255]
img_noise_scale = img_noise / 255

for sigma in sigmas:
    denoised_bm3d = bm3d.bm3d(img_noise_scale, sigma)

    print(psnr(img, denoised_bm3d))

    plt.imshow(denoised_bm3d, cmap='gray')
    plt.show()










