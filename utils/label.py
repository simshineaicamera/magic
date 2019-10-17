#encoding=utf8
'''

'''

import os
from os import path
from utils.consts import *
import sys
import argparse,time
import numpy as np
import cv2
import random
from yolov3.detect import detection
from PIL import Image


from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.utils import save_image
from tqdm import tqdm
from stylize.stylize import input_transform
from stylize.stylize import style_transfer
black_list = []
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class gen_xml():
    def __init__(self,root='annotation'):
        '''
        '''
        self.root_a = ET.Element(root)
        
        self.sub_root_a = None
        self.sub_sub_root_a = None
        
    def set_sub_node(self,last,sub_node,val):#last = root','sub_root' or 'sub_sub_root'
        if last == 'root':
            b = ET.SubElement(self.root_a, sub_node)
            b.text = val
        elif last == 'sub_root':
            b = ET.SubElement(self.sub_root_a, sub_node)
            b.text = val
        elif last == 'sub_sub_root':
            b = ET.SubElement(self.sub_sub_root_a, sub_node)
            b.text = val
            
    def set_sub_root(self,last,sub_root):#last = root','sub_root'
        if last == 'root':
            self.sub_root_a = ET.SubElement(self.root_a, sub_root)
        elif last == 'sub_root':
            self.sub_sub_root_a = ET.SubElement(self.sub_root_a, sub_root)
    def out(self,filename):
        fp = open(filename,'wb')
        tree = ET.ElementTree(self.root_a)
        tree.write(fp)
        fp.close()
def check_bbox(width,height,xmin,ymin,xmax,ymax):
    if xmin < 0:
        xmin = 0
    if xmin > width:
        xmin = width
    if xmax < 0:
        xmax = 0
    if xmax > width:
        xmax = width
    
    if ymin < 0:
        ymin = 0
    if ymin > height:
        ymin = height
    if ymax < 0:
        ymax = 0
    if ymax > height:
        ymax = height
    return xmin,ymin,xmax,ymax
def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames
def auto_label(img, box, name, label, height, width, depth):
    k = 0
    xml_name = name.replace('.jpg','.xml')
    # print(xml_name)
    if not os.path.exists(os.path.join(xml_path,xml_name)):
        my_xml = gen_xml('annotation')
        my_xml.set_sub_node('root','filename','%s'%name)
        my_xml.set_sub_root('root','size')
        my_xml.set_sub_node('sub_root','%s'%'height','%s'%height)
        my_xml.set_sub_node('sub_root','%s'%'width','%s'%width)
        my_xml.set_sub_node('sub_root','%s'%'depth','%s'%depth)
    else:
        #print(xml_name)
        list_need=['filename','size','name','part']
        list_sub_need=['width','height','depth','xmin','ymin','xmax','ymax']
        tree = ET.ElementTree(file=os.path.join(xml_path, xml_name))
        save_person = False
        skip_part = False
        save_size = False
        save_person_n = 0
        save_size_n = 0
        my_xml = gen_xml('annotation')
        for elem in tree.iter():
            if elem.tag in list_need:
                if elem.tag == 'filename':
                    my_xml.set_sub_node('root','filename','%s'%elem.text)
                if elem.tag == 'size':
                    save_size = True
                    my_xml.set_sub_root('root','size')
                if elem.tag == 'name' and elem.text in [label]:
                    my_xml.set_sub_root('root','object')
                    my_xml.set_sub_node('sub_root','name',elem.text)
                    my_xml.set_sub_root('sub_root','bndbox')
                    save_person = True
                if elem.tag == 'part':
                    skip_part = True
                    
            elif elem.tag in list_sub_need:
                if save_size and elem.tag in list_sub_need[:3]:
                    
                    my_xml.set_sub_node('sub_root','%s'%elem.tag,'%s'%elem.text)
                    save_size_n += 1
                    if save_size_n == 3:
                        save_size = False
                        save_size_n = 0
                elif elem.tag in list_sub_need[3:]:
                    save_person_n += 1
                    if skip_part:
                        if save_person_n == 4:
                            skip_part = False
                            save_person_n = 0
                            
                    elif save_person:
                        my_xml.set_sub_node('sub_sub_root','%s'%elem.tag,'%s'%elem.text)
                        if save_person_n == 4:
                            save_person = False
                            save_person_n = 0
                    if save_person_n == 4:
                        save_person_n = 0
        
    xmin,ymin,xmax,ymax = check_bbox(width,height,box[0],box[1],box[2],box[3])
    my_xml.set_sub_root('root','object')
    my_xml.set_sub_node('sub_root','name',label)
    my_xml.set_sub_root('sub_root','bndbox')
    my_xml.set_sub_node('sub_sub_root','%s'%'xmin','%d'%xmin)
    my_xml.set_sub_node('sub_sub_root','%s'%'ymin','%d'%ymin)
    my_xml.set_sub_node('sub_sub_root','%s'%'xmax','%d'%xmax)
    my_xml.set_sub_node('sub_sub_root','%s'%'ymax','%d'%ymax)
    my_xml.out(os.path.join(xml_path,xml_name))

def data_generate(img, box, name, label):
    global black_list
    list_examples = os.listdir(src_path)
    img_list = []
    src_len = len(list_examples)
    i = 0
    while i< multi_num:
        a = random.randint(0, src_len-2)
        if list_examples[a] not in black_list:
            img_list.append(list_examples[a])
            black_list.append(list_examples[a])
        if len(img_list)>3:
            break
        i=i+1

    img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    try:
        img2 = Image.fromarray(img)
    except Exception:
        return
    img_w = int(box[2]-box[0])
    img_h = int(box[3]-box[1])
    if img_w>480:
        img_w = 400
    if img_h >480:
        img_h = 400
    for example in img_list:
        file_name = os.path.join(jpg_path, example)
        if os.path.exists(file_name):
            img1 = Image.open(file_name)
        else:
            img1 = Image.open(os.path.join(src_path, example))
        im_size = img1.size
        xc = random.randint(0,int(480-img_w))
        yc = random.randint(0,int(480-img_h))

        img1.paste(img2,(xc,yc))
        xmin = xc
        xmax = int(xc + img_w)
        ymin = yc
        ymax = int(yc + img_h)
        img1.convert('RGB')
        img1.save(os.path.join(jpg_path,example))
        
        height = im_size[1]
        width = im_size[0]
        depth=3
        auto_label(img1, [xmin, ymin, xmax, ymax], example, label, height, width, depth)
        

def auto_label_yolov3(label, index):
    
    
    list_dir = os.listdir(src_path)
    input_names = []
    for f in list_dir:
        input_names.append(os.path.join(src_path,f))
    #print(len(input_names))
    img_list, bounding_boxes, model_size = detection(0.5, 0.5, input_names)
    print("detection is finished")
    kk = 0
    total = len(bounding_boxes)
    #print(total)
    for kk in range(total):
        
        boxes_dictb = bounding_boxes[kk]
        images = img_list[kk]
        jj = 0
        for jj in range(4):
            #print("jj is ",jj)
            boxes_dict = boxes_dictb[jj]
            img_name = images[jj]
            img = cv2.imread(img_name)
            height = img.shape[0]
            width = img.shape[1]
            depth=img.shape[2]
            f = img_name.split('/')[-1]
            resize_factor = \
                (img.shape[1] / model_size[0], img.shape[0] / model_size[1])
            boxes = boxes_dict[index-1]
            if np.size(boxes) != 0:
                if not os.path.exists(os.path.join(jpg_path,f)):
                    os.system(mv_cmd.format(img_name, jpg_path))
                for box in boxes:
                    xy = box[:4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    x1, y1 = xy[2], xy[3]
                    auto_label(img, [x0,y0,x1,y1], f, label, height, width, depth)
                    data_generate(img, [x0,y0,x1,y1], f, label)
                    
                    #cv2.rectangle(img,(int(box[0]), int(box[1])),(int(box[2]),int(box[3])),(0,255,0), 2)

            # cv2.imshow("AutoLabel", img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # else:
            #     continue

def stylize(img_name, device, vgg, decoder, content_tf, style_tf):
    content_path = os.path.join(jpg_path, img_name)
    content_path = Path(content_path)
    content_img = Image.open(content_path).convert('RGB')
    styles = os.listdir(style_dir)
    # content_tf = input_transform(0, False)
    # style_tf = input_transform(0, False)
    #k = 0 
    for k in range(num_styles):
        tmp = random.randint(0, 99)
        style_path = styles[tmp] 
        try:
            style_img = Image.open(os.path.join(style_dir, style_path)).convert('RGB')
        except OSError as e:
            print('Skipping stylization of %s with %s due to error below' %(content_path, style_path))
            print(e)
            continue
        
        content = content_tf(content_img)
        style = style_tf(style_img)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        alpha = random.randint(5,10)
        alpha = alpha/10
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    alpha)
        output = output.cpu()

        #rel_path = content_path.relative_to(jpg_path)
        #out_dir = output_dir.joinpath(rel_path.parent)

        # create directory structure if it does not exist
        # if not out_dir.is_dir():
        #     out_dir.mkdir(parents=True)

        content_name = content_path.stem
        style_name = style_path.split('.')[0]
        out_filename = content_name + '-stylized-' + style_name + content_path.suffix
        output_name = os.path.join(jpg_path, out_filename)

        save_image(output, output_name, padding=0) #default image padding is 2.
        xml_name_new = content_name + '-stylized-' + style_name + '.xml'
        xml_name_old = content_name + '.xml'
        xml_name_old = os.path.join(xml_path, xml_name_old)
        xml_name_new = os.path.join(xml_path, xml_name_new)
        os.system(copy_command.format(xml_name_old, xml_name_new))
        style_img.close()
        k+=1
        #if k>=num_styles:
         #   break
    content_img.close()


