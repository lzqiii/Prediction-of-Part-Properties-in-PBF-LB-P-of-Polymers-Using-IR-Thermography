import os
import PIL.Image as Image
import numpy as np
import random


def get_all_images(folderpath):
    # Retrieve the jpg file in the path
    # f.endswith（）  restrict file types
    # The return is an absolute path
    return [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith('.png')]


all_folder_name = ['1 P6 41-80', '1 P6 121-160', '1 P6 201-240', '2 P3 281-320', '2 P3 361-400', '2 P3 441-480',
                   '3 P2 521-560', '3 P2 601-640', '3 P2 681-720', '4 P7 761-800', '4 P7 841-880', '4 P7 921-960',
                   '5 P8 1001-1040', '5 P8 1081-1120', '5 P8 1161-1200', '6 P5 1241-1280', '6 P5 1321-1360',
                   '6 P5 1401-1440',
                   '7 P4 1481-1520', '7 P4 1561-1600', '7 P4 1641-1680', '8 P1 1721-1760', '8 P1 1801-1840',
                   '8 P1 1881-1920'
                   ]
layer_num_of_each_folder = [['74', '75', '76', '77', '78', '79'],
                            ['154', '155', '156', '157', '158', '159'],
                            ['233', '234', '235', '236', '237', '238'],
                            ['313', '314', '315', '316', '317', '318'],
                            ['393', '394', '395', '396', '397', '398'],
                            ['473', '474', '475', '476', '477', '478'],
                            ['553', '554', '555', '556', '557', '558'],
                            ['633', '634', '635', '636', '637', '638'],
                            ['713', '714', '715', '716', '717', '718'],
                            ['793', '794', '795', '796', '797', '798'],
                            ['873', '874', '875', '876', '877', '878'],
                            ['953', '954', '955', '956', '957', '958'],
                            ['1033', '1034', '1035', '1036', '1037', '1038'],
                            ['1113', '1114', '1115', '1116', '1117', '1118'],
                            ['1193', '1194', '1195', '1196', '1197', '1198'],
                            ['1273', '1274', '1275', '1276', '1277', '1278'],
                            ['1353', '1354', '1355', '1356', '1357', '1358'],
                            ['1433', '1434', '1435', '1436', '1437', '1438'],
                            ['1513', '1514', '1515', '1516', '1517', '1518'],
                            ['1593', '1594', '1595', '1596', '1597', '1598'],
                            ['1673', '1674', '1675', '1676', '1677', '1678'],
                            ['1753', '1754', '1755', '1756', '1757', '1758'],
                            ['1833', '1834', '1835', '1836', '1837', '1838'],
                            ['1913', '1914', '1915', '1916', '1917', '1918']]
# Name of folders
img_ref = Image.open('E:/Thesis/code/pre_process2/zero.png').convert('L')
i = 0
for folder_name in all_folder_name:
    for layer_num in layer_num_of_each_folder[i]:
        main_path = 'E:/Thesis/code/pre_process2/V12 Chou/' + folder_name + '/' + layer_num
        k = 0
        data = np.empty((2000, 360, 360))
        for path in get_all_images(main_path):
            # Iterate through all images in this folder
            # Read images and convert to grayscale single channel images
            img = Image.open(path).convert('L')
            left = 155
            top = 80
            right = 515
            under = 440
            # Set crop points，to obtain a 200x200 image
            box = (int(left), int(top), int(right), int(under))
            # The input for the box must be int
            img_new = img.crop(box)  # .rotate(90)
            data[k] = np.array(img_new)
            # Store all images in an array
            k = k + 1
            # img_new.save(path)
        real_image_array = Image.fromarray(np.max(data, axis=0))
        # Find the maximum value of the corresponding pixel points in all images
        # real_image_array.show()
        real_image_array = real_image_array.convert('L')
        size_of_img = real_image_array.size
        # real_image_array.save(('E:/Thesis/code/pre_process2/V12 Chou/' + layer_num + '.png'))
        if folder_name in ['1 P6 41-80', '2 P3 281-320', '3 P2 521-560', '4 P7 761-800', '5 P8 1001-1040',
                           '6 P5 1241-1280', '7 P4 1481-1520', '8 P1 1721-1760']:
            background1 = img_ref.copy()
            box1 = (0, 0, 115, size_of_img[1])
            region1 = real_image_array.crop(box1)
            x1 = random.randint(0, 200)
            x3 = random.choice([0, 90, 180, 270])
            background1.paste(region1, (x1, 0))
            background1 = background1.rotate(x3)

            background2 = img_ref.copy()
            box2 = (115, 0, 165, size_of_img[1])
            region2 = real_image_array.crop(box2)
            x1 = random.randint(0, 200)
            x3 = random.choice([0, 90, 180, 270])
            background2.paste(region2, (x1, 0))
            background2 = background2.rotate(x3)

            background3 = img_ref.copy()
            box3 = (165, 0, 215, size_of_img[1])
            region3 = real_image_array.crop(box3)
            x1 = random.randint(0, 200)
            x3 = random.choice([0, 90, 180, 270])
            background3.paste(region3, (x1, 0))
            background3 = background3.rotate(x3)

            background4 = img_ref.copy()
            box4 = (215, 0, 360, size_of_img[1])
            region4 = real_image_array.crop(box4)
            x1 = random.randint(0, 200)
            x3 = random.choice([0, 90, 180, 270])
            background4.paste(region4, (x1, 0))
            background4 = background4.rotate(x3)

            background1.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L1_' + layer_num + '.png'))
            background2.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L2_' + layer_num + '.png'))
            background3.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L3_' + layer_num + '.png'))
            background4.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L4_' + layer_num + '.png'))
        elif folder_name in ['1 P6 201-240', '2 P3 441-480', '3 P2 681-720', '4 P7 921-960', '5 P8 1161-1200',
                             '6 P5 1401-1440', '7 P4 1641-1680', '8 P1 1881-1920']:
            background1 = img_ref.copy()
            box1 = (0, 0, 180, size_of_img[1])
            region1 = real_image_array.crop(box1)
            x1 = random.randint(0, 180)
            x3 = random.choice([0, 90, 180, 270])
            background1.paste(region1, (x1, 0))
            background1 = background1.rotate(x3)

            background2 = img_ref.copy()
            box2 = (180, 0, 220, size_of_img[1])
            region2 = real_image_array.crop(box2)
            x1 = random.randint(0, 200)
            x3 = random.choice([0, 90, 180, 270])
            background2.paste(region2, (x1, 0))
            background2 = background2.rotate(x3)

            background3 = img_ref.copy()
            box3 = (220, 0, 280, size_of_img[1])
            region3 = real_image_array.crop(box3)
            x1 = random.randint(0, 200)
            x3 = random.choice([0, 90, 180, 270])
            background3.paste(region3, (x1, 0))
            background3 = background3.rotate(x3)

            background4 = img_ref.copy()
            box4 = (280, 0, 360, size_of_img[1])
            region4 = real_image_array.crop(box4)
            x1 = random.randint(0, 200)
            x3 = random.choice([0, 90, 180, 270])
            background4.paste(region4, (x1, 0))
            background4 = background4.rotate(x3)
            background1.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L6_' + layer_num + '.png'))
            background2.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L7_' + layer_num + '.png'))
            background3.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L8_' + layer_num + '.png'))
            background4.save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L9_' + layer_num + '.png'))
        else:
            x3 = random.choice([0, 90, 180, 270])
            real_image_array.rotate(x3).save(('E:/Thesis/code/pre_process2/V12 Chou/' + '4L5_' + layer_num + '.png'))
    i = i + 1
