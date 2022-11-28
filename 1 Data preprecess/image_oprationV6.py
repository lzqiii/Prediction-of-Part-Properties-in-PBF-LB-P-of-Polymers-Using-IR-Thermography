import os
import PIL.Image as Image
import numpy as np
import random

def get_all_images(folderpath):
    # Retrieve the jpg file in the path
    # f.endswith（）  restrict file types
    # The return is an absolute path
    return [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith('.png')]


all_folder_name = ['5x5 P1 400-480', '5x5 P2 520-600', '5x5 P3 640-720', '10x10 P1 760-840', '10x10 P2 880-960', '10x10 P3 1000-1080']
layer_num_of_each_folder = [['474', '475', '476', '477', '478', '479'],
                            ['594', '595', '596', '597', '598', '599'],
                            ['714', '715', '716', '717', '718', '719'],
                            ['834', '835', '836', '837', '838', '839'],
                            ['954', '955', '956', '957', '958', '959'],
                            ['1074', '1075', '1076', '1077', '1078', '1079']]
# Name of folders
img_ref = Image.open('E:/Thesis/code/pre_process2/zero.png').convert('L')
i = 0
for folder_name in all_folder_name:
    for layer_num in layer_num_of_each_folder[i]:
        main_path = 'E:/Thesis/code/pre_process2/V6 BA Hadenwang/' + folder_name + '/' + layer_num
        k = 0
        data = np.empty((2000, 360, 360))
        for path in get_all_images(main_path):
            # Iterate through all images in this folder
            # Read images and convert to grayscale single channel images
            img = Image.open(path).convert('L')
            left = 140
            top = 75
            right = 500
            under = 435
            # Set crop points，to obtain a 200x200 image
            box = (int(left), int(top), int(right), int(under))
            # The input for the box must be int
            img_new = img.crop(box).rotate(90)
            data[k] = np.array(img_new)
            # Store all images in an array
            k = k + 1
            # img_new.save(path)
        real_image_array = Image.fromarray(np.max(data, axis=0))
        # Find the maximum value of the corresponding pixel points in all images
        # real_image_array.show()
        real_image_array = real_image_array.convert('L')
        size_of_img = real_image_array.size
        real_image_array.save(('E:/Thesis/code/pre_process2/V6 BA Hadenwang/' + layer_num + '.png'))

        background1 = img_ref.copy()
        box1 = (0, 0, size_of_img[0] // 2, size_of_img[1] // 2)
        region1 = real_image_array.crop(box1)
        x1 = random.randint(0, 180)
        x2 = random.randint(0, 180)
        x3 = random.choice([0, 90, 180, 270])
        background1.paste(region1.rotate(x3), (x1, x2))
        background2 = img_ref.copy()
        box2 = (size_of_img[0] // 2, 0, size_of_img[0], size_of_img[1] // 2)
        region2 = real_image_array.crop(box2)
        x1 = random.randint(0, 180)
        x2 = random.randint(0, 180)
        x3 = random.choice([0, 90, 180, 270])
        background2.paste(region2.rotate(x3), (x1, x2))
        background3 = img_ref.copy()
        box3 = (0, size_of_img[1] // 2, size_of_img[0] // 2, size_of_img[1])
        region3 = real_image_array.crop(box3)
        x1 = random.randint(0, 180)
        x2 = random.randint(0, 180)
        x3 = random.choice([0, 90, 180, 270])
        background3.paste(region3.rotate(x3), (x1, x2))
        background4 = img_ref.copy()
        box4 = (size_of_img[0] // 2, size_of_img[1] // 2, size_of_img[0], size_of_img[1])
        region4 = real_image_array.crop(box4)
        x1 = random.randint(0, 180)
        x2 = random.randint(0, 180)
        x3 = random.choice([0, 90, 180, 270])
        background4.paste(region4.rotate(x3), (x1, x2))
        background1.save(('E:/Thesis/code/pre_process2/V6 BA Hadenwang/' + '4up_left' + layer_num + '.png'))
        background2.save(('E:/Thesis/code/pre_process2/V6 BA Hadenwang/' + '4up_right' + layer_num + '.png'))
        background3.save(('E:/Thesis/code/pre_process2/V6 BA Hadenwang/' + '4bottom_left' + layer_num + '.png'))
        background4.save(('E:/Thesis/code/pre_process2/V6 BA Hadenwang/' + '4bottom_right' + layer_num + '.png'))
        # Divide the image into four parts and rotate the split image so that the parts are on the same corner

    i = i + 1
