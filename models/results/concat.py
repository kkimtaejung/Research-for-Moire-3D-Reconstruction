import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/home/a/Desktop/Deep_Learning/Image_Inpainting/CSA-inpainting-master/CSA_Inpainting/models/results/0_real.bmp',0)

img2 = cv2.imread('/home/a/Desktop/Deep_Learning/Image_Inpainting/CSA-inpainting-master/CSA_Inpainting/models/results/mask.bmp',0)

for i in range(256):
	for j in range(256):
		if img2[i][j] != 0:
			img[i][j] = img2[i][j]

cv2.imwrite('/home/a/Desktop/Deep_Learning/Image_Inpainting/CSA-inpainting-master/CSA_Inpainting/models/results/mask_image.bmp',img)
