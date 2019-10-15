### Magic
<div style="text-align: justify">
The project designed for simplifying training process as much as possible. We have used SOTA tracking and object detection models
for automatic labeling.
### Installation
Requirements: Ubuntu 16.04/18.04 , python>=3.6.
</div>
<br>
If you have already installed [SIMCAM_SDK](https://github.com/simshineaicamera/SIMCAM_SDK) you can skip this step
1. Compile [caffe-ssd](https://github.com/weiliu89/caffe/tree/ssd) on your system
2. Change the caffe path in '$Magic/pretrained/create_lmdb.sh' file 
3. install python packages

Compile caffe-ssd gpu version and set caffe path into create_lmdb.sh file
install requirments.txt for python

Project tested on ubuntu 16.04 and 18.04  with python3.6
