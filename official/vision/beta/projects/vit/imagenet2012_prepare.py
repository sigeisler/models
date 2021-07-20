
import tensorflow_datasets as tfds

DATASET_NAME = 'imagenet2012'
PATH_TO_IMAGENET2012 = '/buckets/raw/'
LOCAL_PATH_TO_DATASETS = '~/tensorflow_datasets'

builder = tfds.builder(DATASET_NAME, data_dir=LOCAL_PATH_TO_DATASETS)

config = tfds.download.DownloadConfig(manual_dir=PATH_TO_IMAGENET2012)
builder.download_and_prepare(download_config=config)
