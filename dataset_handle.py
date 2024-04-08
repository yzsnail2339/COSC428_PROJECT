import xml.etree.ElementTree as ET
import os

def convert(size, box):

    b1, b2, b3, b4 = box
    if b2 > w:
        b2 = w
    if b4 > h:
        b4 = h
    box = (b1, b2, b3, b4)

    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = round(x, 6)
    w = round(w, 6)
    y = round(y, 6)
    h = round(h, 6)
    return x, y, w, h

def convert_annotation(data_dir, image_id, classes):
    in_file = open(os.path.join(data_dir, 'Annotations', '%s.xml' % (image_id)))
    out_file = open(os.path.join(data_dir, 'labels', '%s.txt' % (image_id)), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))

        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


# def convert_VOC_to_YOLO(data_dir):
#     sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
#     classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
#                "bus", "car", "cat", "chair", "cow",
#                "diningtable", "dog", "horse", "motorbike", "person",
#                "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#     wd = os.getcwd()

#     for year, image_set in sets:
#         if not os.path.exists(os.path.join(data_dir, "labels")):
#             os.makedirs(os.path.join(data_dir, "labels"))
#         image_ids = open(os.path.join(data_dir, 'ImageSets/Main', '%s.txt' % (image_set))).read().strip().split()
#         list_file = open(os.path.join(data_dir, '%s_%s.txt' % (year, image_set)), 'w')
#         for image_id in image_ids:
#             list_file.write('%s/JPEGImages/%s.jpg\n' % (wd, image_id))
#             convert_annotation(data_dir, year, image_id)
#         list_file.close()

# # 示例用法：
# data_directory = "/path/to/your/data/directory"  # 更改为实际的数据集目录
# convert_VOC_to_YOLO(data_directory)