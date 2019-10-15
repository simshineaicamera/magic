#-*- coding:utf-8 -*-
# author: sqc
# This .py file realized data augmentation for object detection

import os, cv2, shutil
import imageio 
import imgaug as ia # needs 1.15 numpy
import numpy as np
import xml.dom.minidom as xmldom
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from gen_xml import write_xml
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class DataAug:
    def __init__(self, input_dir, target_dir):
        """
        Initialize the parameter in thi class 
            args:
                input_dir: {'img': the path of image folder, 'xml': the path of .xml folder}
                target_dir: the path of the folder to contain the img folder and xml folder
        """
        self.img_dir = input_dir['img']
        self.xml_dir = input_dir['xml']

        self.img_save_dir = os.path.join(target_dir, 'aug_imgs')
        self.xml_save_dir = os.path.join(target_dir, 'aug_xmls')
        self.make_dirs(target_dir)
    
    def make_dirs(self, target_dir):
        """
        remove the old and make the new target_dir
        """
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            os.makedirs(self.img_save_dir, mode=0o777)
            os.makedirs(self.xml_save_dir, mode=0o777)
        else:
            os.makedirs(self.img_save_dir, mode=0o777)
            os.makedirs(self.xml_save_dir, mode=0o777)


    def parse_xml(self, xml_path):
        """
        get the information in the xml file
            args:
                xml_path: the path to single .xml file
            returns:
                output_dict: {'size':(H,W,C), 
                              'objects':[(label_name,x1,y1,x2,y2),(label_name,x1,y1,x2,y2)]}
        """
        output_dict = {}
        DOMTree = xmldom.parse(xml_path)
        annotation = DOMTree.documentElement
        #img_name = annotation.getElementsByTagName('filename')[0].firstChild.data
        img_size = annotation.getElementsByTagName('size')
        img_height = img_size[0].getElementsByTagName('height')[0].childNodes[0].data
        img_width = img_size[0].getElementsByTagName('width')[0].childNodes[0].data
        img_depth = img_size[0].getElementsByTagName('depth')[0].childNodes[0].data
        output_dict['size'] = (img_height, img_width, img_depth)
        #print(output_dict)

        _objects = annotation.getElementsByTagName('object')
        output_dict['objects'] = list()
        for _object in _objects:
            label_name = _object.getElementsByTagName('name')[0].childNodes[0].data
            #print(label_name)
            bbox = _object.getElementsByTagName('bndbox')[0]
            left = bbox.getElementsByTagName('xmin')[0].childNodes[0].data
            top = bbox.getElementsByTagName('ymin')[0].childNodes[0].data
            right = bbox.getElementsByTagName('xmax')[0].childNodes[0].data
            bottom = bbox.getElementsByTagName('ymax')[0].childNodes[0].data
            res_tuple = (label_name, int(left), int(top), int(right), int(bottom))
            output_dict['objects'].append(res_tuple)
        #print(output_dict) 
        return output_dict # {'size':tuple, 'objects':list}



            
    def plot_augdata(self, img_aug_list, bbs_aug_list):
        """
        plot the augment results
            args:
                img_aug_list: a list contains augmented imgs
                bbs_aug_list: a list contains augmented bbox imgaug objects corresponding to the imgs
                (generally from the self.generate_augdata())
        """
        try:
            assert len(img_aug_list) == len(bbs_aug_list)
        except Exception:
            return False
        img_amount = len(img_aug_list)
        fig_size = 4
        fig = plt.figure(figsize=(fig_size*img_amount,fig_size))
        for i in range(img_amount):
            img_show = bbs_aug_list[i].draw_on_image(img_aug_list[i], size=2, color=[0, 0, 255])
            plot_num = 100 + img_amount*10 + i + 1
            plt.subplot(plot_num)
            plt.imshow(img_show)
        plt.show()




    def generate_augdata(self, img_path, xml_path, SHOW=False):
        """
        Generate augmented data and return the result
            args:
                img_path: the path to a single img
                xml_path: the path to the corresponding image's .xml file
            returns:
                image_aug_list: a list contains augmented images
                bbs_aug_list: a list contains augmented bbox imgaug objects corresponding to the imgs
        """
        # parse the xml file
        object_dict = self.parse_xml(xml_path)

        ori_img_name = os.path.split(img_path)[-1]

        ia.seed(1)
        image = imageio.imread(img_path)
        label_list = [_object[0] for _object in object_dict['objects']]
        bbox_list = [ia.BoundingBox(x1=_object[1], y1=_object[2], x2=_object[3], y2=_object[4], label=_object[0]) \
            for _object in object_dict['objects']]

        # -------- Rotate and Flip start -------- #
        bbs = BoundingBoxesOnImage(bbox_list, shape=image.shape)

        seq_0 = iaa.Sequential([iaa.Fliplr(p=1)])
        image_aug_0, bbs_aug_0 = seq_0(image=image, bounding_boxes=bbs)
        bbs_aug_0 = bbs_aug_0.remove_out_of_image()
        
        seq_1 = iaa.Sequential([iaa.Affine(rotate=90)])
        image_aug_1, bbs_aug_1 = seq_1(image=image, bounding_boxes=bbs)
        bbs_aug_1 = bbs_aug_1.remove_out_of_image()
        
        seq_2 = iaa.Sequential([iaa.Affine(rotate=90), iaa.Fliplr(p=1)])
        image_aug_2, bbs_aug_2 = seq_2(image=image, bounding_boxes=bbs)
        bbs_aug_2 = bbs_aug_2.remove_out_of_image()

        seq_3 = iaa.Sequential([iaa.Affine(rotate=-90)])
        image_aug_3, bbs_aug_3 = seq_3(image=image, bounding_boxes=bbs)
        bbs_aug_3 = bbs_aug_3.remove_out_of_image()

        seq_4 = iaa.Sequential([iaa.Affine(rotate=-90), iaa.Fliplr(p=1)])
        image_aug_4, bbs_aug_4 = seq_4(image=image, bounding_boxes=bbs)
        bbs_aug_4 = bbs_aug_4.remove_out_of_image()
        # ******** Rotate and Flip ends ******** #

        # -------- Rescale and Crop start -------- #
        seq_5 = iaa.Sequential([iaa.Affine(scale=(0.6,0.9))])
        image_aug_5, bbs_aug_5 = seq_5(image=image, bounding_boxes=bbs)
        bbs_aug_5 = bbs_aug_5.remove_out_of_image()

        seq_6 = iaa.Sequential([iaa.Crop(percent=0.25, keep_size=True)]) # set keep_size=True to let the cropped img be amplified to the ori size
        #seq_6 = iaa.Sequential([iaa.Affine(rotate=45)])
        image_aug_6, bbs_aug_6 = seq_6(image=image, bounding_boxes=bbs)
        bbs_aug_6 = bbs_aug_6.remove_out_of_image()
        print(bbs_aug_6.bounding_boxes)
        # ******** Rescale and Crop ends ******** #

        image_aug_list = [image_aug_0, image_aug_1, image_aug_3, image_aug_5, image_aug_6]
        bbs_aug_list = [bbs_aug_0, bbs_aug_1, bbs_aug_3, bbs_aug_5, bbs_aug_6]
        if SHOW: self.plot_augdata(image_aug_list[-1:], bbs_aug_list[-1:])
        image_aug_list.insert(0, image)
        bbs_aug_list.insert(0, bbs)
        return image_aug_list, bbs_aug_list

    def save_augdata(self, img_aug_list, bbs_aug_list, **img_info):
        """
        process the augmented data and save the .xml file
            args: 
                image_aug_list: a list contains augmented images
                bbs_aug_list: a list contains augmented bbox imgaug objects corresponding to the imgs
                (These two arguments above are from the self.generate_augdata())
                img_info: argument dict, {'img_name':original image name, 'img_size':original image size}
            returns:
                If saved successfully: return True, else False

        """
        def save_img(img_name, img_file):
            """
            save the imgs from augmentation
            """
            if os.path.splitext(img_name)[-1] in ['.jpg', '.jpeg', '.png', '.bmp']:
                save_path = os.path.join(self.img_save_dir, img_name)
                imageio.imsave(save_path, img_file)
                return save_path
            else:
                return False

        try:
            assert len(img_aug_list) == len(bbs_aug_list)
        except Exception:
            return False

        ori_image_name = img_info.get('img_name', None)
        ori_image_size = img_info.get('img_size', None)

        aug_data_num = len(img_aug_list)
        ori_pre, ori_ext = os.path.splitext(ori_image_name)
        for ELE_INDEX in range(aug_data_num): # for each augmented img and bboxes
            bbs_aug = bbs_aug_list[ELE_INDEX]
            bboxes_aug, label_list = [], []

            img_savename = ori_pre+'_{}'.format(ELE_INDEX)+ori_ext
            img_savefile = img_aug_list[ELE_INDEX]
            img_savepath = save_img(img_savename, img_savefile)
            xml_savename = ori_pre+'_{}'.format(ELE_INDEX)+'.xml'
            xml_save_path = os.path.join(self.xml_save_dir, xml_savename)

            for BBOX_INDEX in range(len(bbs_aug.bounding_boxes)):
                bboxes = bbs_aug.bounding_boxes[BBOX_INDEX]
                left, top, right, bottom = bboxes.x1_int, bboxes.y1_int, bboxes.x2_int, bboxes.y2_int
                if left < 0 or top < 0 or right > int(ori_image_size[1]) or bottom > int(ori_image_size[0]): # do not include the bbox that overstep the boundary
                    print(left, top, right, bottom, int(ori_image_size[1]), int(ori_image_size[0]))
                    continue 
                label = bboxes.label
                bbox_aug = [left, top, right, bottom]
                bboxes_aug.append(bbox_aug)
                label_list.append(label)
            write_xml(img_path=img_savepath, bboxes=bboxes_aug, labels=label_list, out_path=xml_save_path)
        return True

    def run_aug(self):
        """
        run the augment code and save the outputs
        """
        img_dir, xml_dir = self.img_dir, self.xml_dir
        xml_names = os.listdir(xml_dir)
        img_names = os.listdir(img_dir)
        for IMG_INDEX in tqdm(range(len(img_names))):
            img_name = img_names[IMG_INDEX]
            img_xml_name = os.path.splitext(img_name)[0]+'.xml' # the corresponding xml file name
            if img_xml_name not in xml_names:
                continue
            print(img_name)
            image_aug_list, bbs_aug_list = self.generate_augdata(os.path.join(img_dir, img_name), 
                                os.path.join(xml_dir, img_xml_name)) # generate augment data
            object_dict = self.parse_xml(os.path.join(xml_dir, img_xml_name))
            if self.save_augdata(image_aug_list, bbs_aug_list, 
                                    **{'img_name':img_name, 'img_size':object_dict['size']}) == False:
                print('----Save failed, image name is: {}----'.format(img_name))

def main():
    img_dir = './JPEGImages' # The input image folder's path
    xml_dir = './Annotations' # The input xml folder's path
    save_dir = './augmented' # The output folder which contains .xml's folder and img's folder
    # img_dir = './test_aug_img'
    # xml_dir = './test_aug_xml' 
    DA = DataAug(input_dir={'img':img_dir,'xml':xml_dir}, target_dir=save_dir)
    DA.run_aug()

if __name__ == '__main__':
    main()


