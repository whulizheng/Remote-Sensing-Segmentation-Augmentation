import os
from tqdm import tqdm
from PIL import Image
import numpy as np

pic_path = r"C:\Users\WhuLi\OneDrive\code\python\Remote-Sensing-Segmentation-Augmentation\data\HRSC2016\training\\"
output_path = r"C:\Users\WhuLi\OneDrive\code\python\Remote-Sensing-Segmentation-Augmentation\data\HRSC2016\boats\\"


def scan_file(file_dir, max_num=None):
    if max_num:
        count = 0
        files = []
        for roo, dirs, file in os.walk(file_dir):
            files.append(file)
            count += 1
            if count >= max_num:
                break
        return files[0][0:max_num]
    else:
        files = []
        for roo, dirs, file in os.walk(file_dir):
            files.append(file)
        return files[0]

def clean_pix(imga):
    img_array = np.array(imga)
    shape = img_array.shape
    height = shape[0]
    width = shape[1]
    dst = np.zeros((height, width, 3))
    for h in range(0, height):
        for w in range(0, width):
            (b, g, r) = img_array[h, w]
            if np.linalg.norm(np.array([b, g, r])-np.array([255, 255, 255])) < 120:
                img_array[h, w] = (0, 0, 0)
            elif np.linalg.norm(np.array([b, g, r])-np.array([0, 0, 0])) < 120:
                pass
            else:
                img_array[h, w] = (255, 0, 0)
            dst[h, w] = img_array[h, w]
    return Image.fromarray(np.uint8(dst))




def cut_img(img, x_1, x_2, y_1, y_2):
    return img.crop((x_1, y_1, x_2, y_2))


def cut_resize(img,  x_1, x_2, y_1, y_2, shape=256):
    img = img[0:shape][0:shape]
    boat_img = cut_img(img, x_1, x_2, y_1, y_2)
    return boat_img


if __name__ == "__main__":
    pics = scan_file(pic_path)
    for i in tqdm(range(len(pics))):

        img = Image.open(pic_path+pics[i])
        img = clean_pix(img)
        img.save(output_path+pics[i])
