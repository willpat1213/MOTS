import os
import cv2
from tqdm import tqdm
import numpy as np

mask_folder_path = ""

pixel_counts = {}

# for filename in tqdm(os.listdir(mask_folder_path)):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         img = cv2.imread(os.path.join(mask_folder_path, filename), cv2.IMREAD_GRAYSCALE)

#         unique_values, counts = np.unique(np.array(img), return_counts=True)
#         if 2 in unique_values:
#             print(filename)

#         for value, count in zip(unique_values, counts):
#             if value not in pixel_counts:
#                 pixel_counts[value] = 0
#             pixel_counts[value] += count


textseg_pixel_counts = {0: 2858081934, 1: 244768177}
cocotext_pixel_counts = {0: 3252601976, 1: 13236081}
totaltext_pixel_counts = {0: 1231774387, 1: 40271157}
pixel_counts = cocotext_pixel_counts
print(pixel_counts)
counts = np.array(list(pixel_counts.values()))
log_counts = np.log1p(counts)
weights = 2 * log_counts / np.sum(log_counts)
# l1_norm = np.linalg.norm(log_counts, ord=1)
# weights = log_counts / l1_norm
for class_name, weight in zip(pixel_counts.keys(), weights):
    print(f"Class: {class_name}, Weight: {weight}")