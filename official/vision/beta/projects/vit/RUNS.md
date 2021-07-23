# TPU

Start: `ctpu up --tpu-only --tpu-size=v3-8` or `ctpu up --tpu-only --tpu-size=v3-32`
Stop: `ctpu pause --tpu-only`

Mount buckets: `gcsfuse jaeyounkim-simongeisler /buckets`

# Base

ViT base:
```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {global_batch_size: 16, input_path: '', tfds_name: imagenet2012, tfds_split: train}, validation_data: {global_batch_size: 16, input_path: '', tfds_name: imagenet2012, tfds_split: validation}}}" \
  --model_dir=/home/simongeisler/models/official/vision/beta/projects/vit/runs/debug
```

DEIT base:
```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=deit_imagenet_pretrain_noaug \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {global_batch_size: 16, input_path: '', tfds_name: imagenet2012, tfds_split: train}, validation_data: {global_batch_size: 16, input_path: '', tfds_name: imagenet2012, tfds_split: validation}}}" \
  --model_dir=/home/simongeisler/models/official/vision/beta/projects/vit/runs/debug
```

# Runs

# 01 - VIT: Check how it goes

_Unfortunately it was without dropout and with random augment..._
```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://jaeyounkim-simongeisler}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://jaeyounkim-simongeisler}}}" \
  --model_dir=gs://jaeyounkim-simongeisler/runs/vit-01 |& tee -a /buckets/runs/vit-01/log.txt
```

# 02 - VIT: Check how it goes

```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://jaeyounkim-simongeisler}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://jaeyounkim-simongeisler}}}" \
  --model_dir=gs://jaeyounkim-simongeisler/runs/vit-02 |& tee -a /buckets/runs/vit-02/log.txt
```

# VIT- 03: Check how it goes
```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://jaeyounkim-simongeisler}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://jaeyounkim-simongeisler}}}" \
  --model_dir=gs://jaeyounkim-simongeisler/runs/vit-03 |& tee -a /buckets/runs/vit-03/log.txt
```

# DEIT-01: Check how it goes
```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=deit_imagenet_pretrain_noaug \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://jaeyounkim-simongeisler}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://jaeyounkim-simongeisler}}}" \
  --model_dir=gs://jaeyounkim-simongeisler/runs/deit-01-noaug |& tee -a /buckets/runs/deit-01-noaug/log.txt
```
