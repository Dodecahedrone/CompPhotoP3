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



    # part 3

def bilateral_weight(x1, y1, x2, y2, sigma_s, sigma_i):
    spatial = gaussian_weighting(x1,y1,x2,y2, sigma_s)
    intensity = np.exp(-((img_noise[x1, y1] - img_noise[x2, y2]) ** 2) / (2 * sigma_i ** 2))
    return spatial * intensity

window_radius = 12     
spacial_sigma = [1, 5, 25]
intensity_sigma = [1, 5, 25, 50, 500]

for sigma_s in spacial_sigma:
    for sigma_i in intensity_sigma:
        denoised_bilat = np.zeros_like(img_noise, dtype=float)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                numerator = 0.0
                denominator = 0.0
                for w1 in range(-12,13):
                    for w2 in range(-12,13):
                        if 0 <= i + w1 < img.shape[0] and 0 <= j + w2 < img.shape[1]:
                            weight = bilateral_weight(i, j, i + w1, j + w2, sigma_s, sigma_i)
                            numerator += img_noise[i + w1, j + w2] * weight
                            denominator += weight
                denoised_bilat[i, j] = numerator / denominator

        print(f"σ_s={sigma_s:>2}, σ_i={sigma_i:>3}  PSNR: {psnr(img, denoised_bilat):.2f} dB")
        plt.imshow(denoised_bilat, cmap='gray')
        plt.title(f"σs={sigma_s}, σi={sigma_i}")
        plt.axis('off')
        plt.show()

# part 4

for h_val in [1, 5, 25, 50, 500]:
        den_uint8 = cv2.fastNlMeansDenoising(img_noise.astype(np.uint8),h=h_val,templateWindowSize=7,searchWindowSize=25)
        den = den_uint8.astype(np.float32)
        plt.imshow(den, cmap="gray")
        plt.show()
        print(f"NLM h={h_val:>3}: {psnr(img, den):.2f} dB")


# part 5
sigmas = [5 / 255, 20 / 255, 50 / 255]
img_noise_scale = img_noise / 255

for sigma in sigmas:
    denoised_bm3d = bm3d.bm3d(img_noise_scale, sigma)

    print(psnr(img, denoised_bm3d))

    plt.imshow(denoised_bm3d, cmap='gray')
    plt.show()
