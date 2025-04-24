import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def gaussian_weighting(p1, p2, sigma):
    return np.exp(-0.5 * ((p1-p2) ** 2) / (sigma ** 2))

denoised = np.zeros(img_noise.shape, dtype=float)

for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        numerator = 0.0
        denominator = 0.0

        for w1 in range(-12, 13):
            for w2 in range(-12, 13):
                if not(w1 + i < 0 or w1 + i >= img.shape[0] or w2 + j < 0 or w2 + j >= img.shape[1]):
                    #print(w1 + i, w2 + j)
                    weight = gaussian_weighting(img_noise[i][j], img_noise[i + w1][j + w2], 5)
                    numerator += img_noise[i + w1][j + w2] * weight
                    denominator += weight

        #print(numerator, denominator)
        denoised[i][j] = numerator/denominator

plt.imshow(denoised, cmap='gray')
plt.show()









