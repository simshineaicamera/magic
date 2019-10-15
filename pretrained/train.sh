#!/bin/sh
if ! test -f pretrained/train.prototxt ;then
	echo "error: train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
LOG=log/train.log
mkdir -p snapshot
/home/inomjon/Projects/caffe-ssd/build/tools/caffe train -solver="pretrained/solver_train.prototxt" \
-weights="pretrained/deploy.caffemodel"  \
-gpu 0 2>&1 | tee $LOG



