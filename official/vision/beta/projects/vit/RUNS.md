# TPU

Start: `ctpu up --tpu-only --tpu-size=v3-8`
Stop: `ctpu pause`

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
  --experiment=deit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {global_batch_size: 16, input_path: '', tfds_name: imagenet2012, tfds_split: train}, validation_data: {global_batch_size: 16, input_path: '', tfds_name: imagenet2012, tfds_split: validation}}}" \
  --model_dir=/home/simongeisler/models/official/vision/beta/projects/vit/runs/debug
```

# Runs

# 01 - DEIT: Check how it goes
```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {global_batch_size: 1024, input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://jaeyounkim-simongeisler}, validation_data: {global_batch_size: 1024, input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://jaeyounkim-simongeisler}}}" \
  --model_dir=gs://jaeyounkim-simongeisler/runs/01
```

# 02 - DEIT: Check how it goes
```
PYTHONPATH=/home/simongeisler/models python3 /home/simongeisler/models/official/vision/beta/projects/vit/train.py \
  --experiment=deit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=jaeyounkim-simongeisler-1 \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://jaeyounkim-simongeisler}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://jaeyounkim-simongeisler}}}" \
  --model_dir=gs://jaeyounkim-simongeisler/runs/02
```
