import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('seg_0000.pgm', -1)

rgb_image = cv2.imread('2009_002993.jpg')

numbers = np.unique(image)

#print numbers
#print np.where(numbers == 12)[0][0]

# dilate the segment r pixels, and subtract itself to find touching

kernel = np.ones((3, 3), np.uint8)

bbox = []
adjacent = []


for i in numbers:
    # bool to int: astype(int) or A*1
    region_i = (image == i) * 1.0
    region_index = np.where(region_i == 1.0)

    bbox.append([(np.min(region_index[1]), np.min(region_index[0])),
                 (np.max(region_index[1]), np.max(region_index[0]))])

    dilation = cv2.dilate(region_i.astype(np.float32), kernel, iterations=1)
    region_overlap = dilation - region_i
    flag_overlap = np.unique(region_overlap * image)

    adjacent.append(flag_overlap)


nums_region = len(bbox)

for i in range(0, nums_region):
    current_region = numbers[i]
    x1 = (int(bbox[i][0][0]) + int(bbox[i][1][0]))/2
    y1 = (int(bbox[i][0][1]) + int(bbox[i][1][1]))/2

    adjacent_regions = adjacent[i][1:]

    for j in adjacent_regions:
        adjacent_region_index = np.where(numbers == j)[0][0]

        x2 = (int(bbox[adjacent_region_index][0][0]) + int(bbox[adjacent_region_index][1][0])) / 2
        y2 = (int(bbox[adjacent_region_index][0][1]) + int(bbox[adjacent_region_index][1][1])) / 2

        cv2.line(rgb_image, (x1, y1), (x2, y2), 255, 1)


cv2.imwrite('/home/liuhy/workspace/instance_segmentation/gSLICr/line.png', rgb_image)
