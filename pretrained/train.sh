#!/bin/sh
if ! test -f pretrained/train.prototxt ;then
	echo "error: train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
root_dir="$(pwd)"
echo $root_dir
cd $PYTHONPATH
cd ..
caffe_root="$(pwd)"
echo $caffe_root
cd $root_dir
LOG=log/train.log
mkdir -p snapshot
/home/inomjon/Projects/MobileNet-YOLO/build/tools/caffe train -solver="pretrained/solver.prototxt" \
#-weights="pretrained/deploy.caffemodel"  \
#-gpu 0 2>&1 | tee $LOG



