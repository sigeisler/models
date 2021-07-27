# TPU

Start: `ctpu up --tpu-only --tf-version=nightly --tpu-size=v3-8` or `ctpu up --tpu-only --tf-version=nightly --tpu-size=v3-32`
Stop: `ctpu pause --tpu-only`

Mount buckets: `gcsfuse jaeyounkim-simongeisler /buckets`

# Env

```
export DATA=gsoc-21-vit/tensorflow_datasets
export RUNS=gsoc-21-vit/runs
export TPU=gsoc21-vit-vm
export PYTHONPATH=/home/simon/models
```

```
export DATA=gsoc-21-vit/tensorflow_datasets
export RUNS=gsoc-21-vit/runs
export TPU=jaeyounkim-simongeisler-1
export PYTHONPATH=/home/simongeisler/models
```

# Runs

# 01 - VIT: Check how it goes

_Unfortunately it was without dropout and with random augment..._
```
python3 train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=$RUNS/vit-01 |& tee -a /buckets/runs/vit-01/log.txt
```

# 02 - VIT: Check how it goes

```
python3 train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/vit-02 |& tee -a /buckets/runs/vit-02/log.txt
```

# VIT- 03: Check how it goes
```
python3 train.py \
  --experiment=vit_imagenet_pretrain \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/vit-03 |& tee -a /buckets/runs/vit-03/log.txt
```

# VIT- 04: DEIT init
```
python3 train.py \
  --experiment=vit_imagenet_pretrain_deitinit \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/vit-04-deitinit |& tee -a /buckets/runs/vit-04-deitinit/log.txt
```

# DEIT-01: Check how it goes
```
python3 train.py \
  --experiment=deit_imagenet_pretrain_noaug \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/deit-01-noaug |& tee -a /buckets/runs/deit-01-noaug/log.txt
```

```
python3 train.py \
  --experiment=deit_imagenet_pretrain_sd_randa_erase_repa \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/deit-03-sd-randa-erase-repa |& tee -a /buckets/runs/deit-03-sd-randa-erase-repa/log.txt
```

```
python3 train.py \
  --experiment=deit_imagenet_pretrain_noaug_sd \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/deit-02-noaug-sd |& tee -a /buckets/runs/deit-02-noaug-sd/log.txt
```

```
python3 train.py \
  --experiment=deit_imagenet_pretrain_noaug_sd_erase \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/deit-03-noaug-sd-erase |& tee -a /buckets/runs/deit-03-noaug-sd-erase/log.txt
```

```
python3 train.py \
  --experiment=deit_imagenet_pretrain_noaug_sd_erase_randa \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/deit-04-noaug-sd-erase-randa |& tee -a /buckets/runs/deit-04-noaug-sd-erase-randa/log.txt
```

```
python3 train.py \
  --experiment=deit_imagenet_pretrain_sd_randa_erase_repa \
  --mode train_and_eval \
  --tpu=$TPU \
  --params_override="{runtime: {distribution_strategy: tpu}, task: {train_data: {input_path: '', tfds_name: imagenet2012, tfds_split: train, tfds_data_dir: gs://$DATA}, validation_data: {input_path: '', tfds_name: imagenet2012, tfds_split: validation, tfds_data_dir: gs://$DATA}}}" \
  --model_dir=gs://$RUNS/deit-05-sd-randa-erase-repa |& tee -a /buckets/runs/deit-05-sd-randa-erase-repa/log.txt
```
TODO: rename folder `deit-03-sd-randa-erase` to `deit-05-sd-randa-erase-repa`