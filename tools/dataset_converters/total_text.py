import os
import cv2
import numpy as np
from tqdm import tqdm

# 定义输入和输出文件夹路径
input_folder = ""
output_folder = ""

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# convert_ID = {
#     1: 2,
#     0: 1,
#     255: 0
# }

# 遍历输入文件夹中的所有图像文件
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        thresholded_image = (img > 127).astype(int)

        filename = filename.replace('jpg', 'png')
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, thresholded_image)
    else:
        print(filename)