root_dir="$(pwd)"
echo $root_dir
cd $PYTHONPATH
cd ..
caffe_root="$(pwd)"
pwd
cd $root_dir/data/lmdb_files
bash create_list.sh $caffe_root
bash create_data.sh $caffe_root

