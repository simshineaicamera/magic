# -*- coding: utf-8 -*-


numFramePerSecond = 10  # number of frames extract images
save_images = "data/Images_xmls/images" #  save extracted images 
video_path = "data/Images_xmls/videos/video.mp4"  # keep video file into this folder
video_name = "video.mp4"

label_path = "data/labels/coco.names"
labelmap_path = "data/lmdb_files/labelmap.prototxt"
labelmap_text = '''item {
    name: "none_of_the_above"
    label: 0
    display_name: "background"
}
item {'''
rm_cmd = 'rm {}/*'
mv_cmd = 'mv {} {}'
rm_file = 'rm {}'
copy_command = "cp {} {}"
src_path = 'data/Images_xmls/images'
jpg_path = 'data/Images_xmls/JPEGImages'
xml_path = 'data/Images_xmls/Annotations'
start_train = "bash pretrained/train.sh"
create_lmdb = 'bash pretrained/create_lmdb.sh'
train_percent = 0.9  
root_dir = "data/Images_xmls/"
ftest = open(root_dir+'ImageSets/Main/test.txt', 'w')  
ftrain = open(root_dir+ 'ImageSets/Main/train.txt', 'w') 
LogPath = 'log/train.log'
multi_num = 10
progres_text = 'Epoch: {}, accuracy: {}'
min_examples = 100
style_dir = 'stylize/style_dir'
crop = 'store_true'