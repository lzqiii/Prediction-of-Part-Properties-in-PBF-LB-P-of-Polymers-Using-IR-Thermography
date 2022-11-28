import cv2
import os
"""This code is used to calculate the mean and standard deviation used for normalization"""

means, stdevs = [], []
img_list = []
# Put all the images in this folder
imgs_path = 'E:/Thesis/code/final/1/'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path, item), 0)
    img = img[:, :, np.newaxis]
    img_list.append(img)
    i += 1
    #print(i, '/', len_)
imgs = np.concatenate(img_list, axis=2)
imgs = imgs.astype(np.float32) / 255.
pixels = imgs[:, :, :].ravel()  # 拉成一行

print("normMean = {}".format(np.mean(pixels)))
print("normStd = {}".format(np.std(pixels)))
