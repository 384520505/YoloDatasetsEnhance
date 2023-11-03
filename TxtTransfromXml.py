import os
import cv2
from tqdm import tqdm
from lxml.etree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString
import numpy as np

out_root = r'datasets/tea/labels' # xml文件存放路径
def build_dir(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir


def get_root_lst(root, suffix='jpg', suffix_n=3):
    root_lst, name_lst = [], []

    for dir, file, names in os.walk(root):
        root_lst = root_lst + [os.path.join(dir, name) for name in names if name[-suffix_n:] == suffix]
        name_lst = name_lst + [name for name in names if name[-suffix_n:] == suffix]

    return root_lst, name_lst


def read_txt(path):
    txt_info_lst = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            txt_info_lst.append(list(line.strip('\n').split()))
    txt_info_lst = np.array(txt_info_lst)
    return txt_info_lst


def product_xml(name_img, boxes, codes, img=None, wh=None):
    '''
    :param img: 以读好的图片
    :param name_img: 图片名字
    :param boxes: box为列表
    :param codes: 为列表
    :return:
    '''
    if img is not None:
        width = img.shape[0]
        height = img.shape[1]
    else:
        assert wh is not None
        width = wh[0]
        height = wh[1]
    # print('xml w:{} h:{}'.format(width,height))

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = name_img  # 图片名字

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i, code in enumerate(codes):
        box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = code
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box[0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box[1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box[3]))

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)

    name = name_img[:-4] + '.xml'

    tree = ElementTree(node_root)

    # print('name:{},dom:{}'.format(name, dom))
    return tree, name


def yolov5txt2xml(root_data, txt_root, gt_labels=None, out_dir=None):
    # 获得图像与txt的路径与名称的列表
    img_roots_lst, img_names_lst = get_root_lst(root_data, suffix='jpg', suffix_n=3)
    txt_roots_lst, txt_names_lst = get_root_lst(txt_root, suffix='txt', suffix_n=3)
    # 创建保存xml的文件
    out_dir = build_dir(out_dir) if out_dir is not None else build_dir(os.path.join(out_root, 'val_xml')) # val_xml xml文件存放的文件夹

    label_str_lst = []
    # 通过图像遍历
    for i, img_root in tqdm(enumerate(img_roots_lst)):
        # 获得图像名称，并得到对应txt名称
        img_name = img_names_lst[i]
        txt_name = img_name[:-3] + 'txt'

        if txt_name in txt_names_lst:  # 通过图像获得txt名称是否存在，存在则继续，否则不继续
            txt_index = list(txt_names_lst).index(str(txt_name))  # 获得列表txt对应索引，以便后续获得路径
            # 通过图像获得图像高与宽
            img = cv2.imread(img_root)
            height, width = img.shape[:2]
            # 读取对应txt的信息
            txt_info = read_txt(txt_roots_lst[txt_index])

            # 以下获得txt信息，并保存labels_lst与boxes_lst中，且一一对应
            labels_lst, boxes_lst = [], []
            for info in txt_info:
                label_str = str(info[0])
                if label_str not in label_str_lst:
                    label_str_lst.append(label_str)

                x, y, w, h = float(info[1]) * width, float(info[2]) * height, float(info[3]) * width, float(
                    info[4]) * height
                xmin, ymin, xmax, ymax = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                labels_lst.append(label_str)
                boxes_lst.append([xmin, ymin, xmax, ymax])
            # 是否转换信息
            if gt_labels:  # gt_labels需要和txt类别对应
                labels_lst = [gt_labels[int(lb)] for lb in labels_lst]
            # 构建xml文件
            if len(labels_lst) > 0:
                tree, xml_name = product_xml(img_name, boxes_lst, labels_lst, wh=[w, h])
                tree.write(os.path.join(out_dir, xml_name))

    print('gt label:', gt_labels)
    print('txt label:', label_str_lst)
    print('save root:', out_dir)


if __name__ == '__main__':
    # Volumes/ACASIS_Media/DeepLearning/ultralytics/datasets/preson
    root_path = r'datasets/tea/images/val'  # 图片路径
    txt_root = r'datasets/tea/labels/val'   # txt路径

    gt_labels = ['tea']

    yolov5txt2xml(root_path, txt_root, gt_labels=gt_labels)
