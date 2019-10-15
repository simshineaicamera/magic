import os
import xml.etree.ElementTree as ET
import PIL 
import numpy as np

def create_labeldict(bboxes, labels):
    label_dicts = []
    for label, bbox in zip(labels, bboxes):
        label_dict = {
        'name': label,
        'difficult': '0',
        'xmin': bbox[0], 
        'ymin': bbox[1], 
        'xmax': bbox[2], 
        'ymax': bbox[3]
        }
        label_dicts.append(label_dict)
    return label_dicts

def write_xml(img_path, bboxes, labels, out_path):                     
    label_dicts = create_labeldict(bboxes, labels)
    img_height, img_width, img_depth = np.array(PIL.Image.open(img_path)).shape
    img_name = os.path.basename(img_path)

    root = ET.Element('annotation')                             # create root node Annotation 
    ET.SubElement(root, 'filename').text = img_name             # create child node filename 
    ET.SubElement(root, 'path').text = os.path.realpath(img_path)                 # create child node path 
    
    sources = ET.SubElement(root,'source') 
    ET.SubElement(sources, 'database').text = 'Unknown'

    sizes = ET.SubElement(root,'size')                          # create child node size          
    ET.SubElement(sizes, 'width').text = str(img_width)                 
    ET.SubElement(sizes, 'height').text = str(img_height)
    ET.SubElement(sizes, 'depth').text = str(img_depth) 

    ET.SubElement(root, 'segmented').text = '0'

    for label_dict in label_dicts:
        objects = ET.SubElement(root, 'object')                 # create child node object
        ET.SubElement(objects, 'name').text = label_dict['name']                                                                          
        ET.SubElement(objects, 'pose').text = 'Unspecified'
        ET.SubElement(objects, 'truncated').text = '0'
        ET.SubElement(objects, 'difficult').text = '0'
        bndbox = ET.SubElement(objects, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(label_dict['xmin']))
        ET.SubElement(bndbox, 'ymin').text = str(int(label_dict['ymin']))
        ET.SubElement(bndbox, 'xmax').text = str(int(label_dict['xmax']))
        ET.SubElement(bndbox, 'ymax').text = str(int(label_dict['ymax']))
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding='utf-8')