export BUCKET=gsoc-21-vit
export REMOTE_RAW_DATA=imagenet2012
export LOCAL_DATA_DIR=${HOME}/data/imagenet2012

gsutil cp gs://$BUCKET/$REMOTE_RAW_DATA/ILSVRC2012_img_train.tar $LOCAL_DATA_DIR/ILSVRC2012_img_train.tar
gsutil cp gs://$BUCKET/$REMOTE_RAW_DATA/ILSVRC2012_img_val.tar $LOCAL_DATA_DIR/ILSVRC2012_img_val.tar