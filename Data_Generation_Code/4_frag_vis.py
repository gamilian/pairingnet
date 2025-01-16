import os
import cv2
import numpy as np
from glob import glob

# 在原图上可视化显示切割的碎片

def vis(dir_path,overall_folder):
    base_name = os.path.basename(dir_path)
    if base_name == "00074":
        print(1)
    img_path = list(glob(dir_path + '/*.png'))
    ori_img = cv2.imread(dir_path+"/image.jpg", cv2.IMREAD_GRAYSCALE)
    height, width = ori_img.shape[:2]
    gt_path = os.path.join(dir_path, 'gt.txt')
    bg_path = os.path.join(dir_path, 'bg_color.txt')
    gt_list = []
    with open(gt_path, 'r') as f:
        while True:
            element = f.readline()
            element = element.strip()
            if element == '':
                break
            elif len(element) < 4:
                continue
            else:
                element = element.split()
                element = list(map(str, element))
                gt_list.append(np.array(element, dtype=float).reshape(-1, 3))
    with open(bg_path, 'r') as f:
        bg = f.readline()
    bg = np.asarray(bg.split(), dtype=int)

    empty = np.zeros((height, width, 3), dtype=np.uint8)
    for i, p in enumerate(img_path):
        img = cv2.imread(p)
        # img = cv2.imread(p, cv2.IMREAD_UNCHANGED)

        mask = (img == bg).all(axis=-1)

        img[mask] = (0, 0, 0)
        gray = np.ones(img.shape[:2], dtype=np.uint8)
        gray[~mask] = 255
        _, b_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contour, hierarchy = cv2.findContours(b_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if isinstance(contour, tuple):
            tar = contour[0]
            for va in contour:
                if len(va) > len(tar):
                    tar = va
            contour = tar.reshape(-1, 2)
        else:
            contour = np.asarray(contour, dtype=np.float).reshape(-1, 2)
        for m in range(len(contour)):
            cv2.circle(img, tuple(contour[m].astype(np.int)), 2, (255, 255, 255), -1)

        gt_pose = gt_list[i]
        gt_pose = np.linalg.inv(gt_pose)
        img = cv2.warpAffine(img, gt_pose[:2], (width, height))
        empty += img

    cv2.imwrite(dir_path + '/recover.jpg', empty)
    cv2.imwrite(overall_folder + '/' + base_name + ".jpg", empty)


if __name__ == '__main__':
    import sys
    # fragment path
    overall_folder = sys.argv[1] + "/vis"
    root = sys.argv[2] + "/fragments"
    print(root)
    print(overall_folder)
    parent_path = os.path.dirname(root)
    if os.path.exists(overall_folder) is False:
        os.mkdir(overall_folder)
    case_list = list(glob((root + '/*')))
    for i, case in enumerate(case_list):
        vis(case,overall_folder)

