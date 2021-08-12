try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from PIL import Image
xml_path = r"C:\Users\WhuLi\OneDrive\code\python\Remote-Sensing-Segmentation-Augmentation\data\HRSC2016\Annotations\\"
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


def load_xml(path):
    tree = ET.ElementTree(file=path)
    root = tree.getroot()
    return root


def get_boat_coordinate(xml_path):
    co = []
    root = load_xml(xml_path)
    objects = list(root.iterfind("HRSC_Objects/HRSC_Object"))
    for o in objects:
        tmp = []
        tmp.append(int(list(o.iterfind("box_xmin"))[0].text))
        tmp.append(int(list(o.iterfind("box_xmax"))[0].text))
        tmp.append(int(list(o.iterfind("box_ymin"))[0].text))
        tmp.append(int(list(o.iterfind("box_ymax"))[0].text))
        co.append(tmp)
    return co


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
        co = get_boat_coordinate(xml_path+pics[i].split(".")[0]+".xml")
        for c in range(len(co)):
            img_x = cut_resize(img, co[c][0], co[c][1], co[c][2], co[c][3])
            img_x.save(output_path+str(i)+"_"+str(c)+".png")
