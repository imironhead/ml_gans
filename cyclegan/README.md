# Replicated "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial"

## Results:

### Horse to Zebra
![](/assets/cyclegan_ok_h2z_00.png)
![](/assets/cyclegan_ok_h2z_01.png)

![](/assets/cyclegan_ng_h2z_00.png)
![](/assets/cyclegan_ng_h2z_01.png)

### Zebra to Horse
![](/assets/cyclegan_ok_z2h_00.png)
![](/assets/cyclegan_ok_z2h_01.png)

![](/assets/cyclegan_ng_z2h_00.png)
![](/assets/cyclegan_ng_z2h_01.png)

### Image to Image Translation?
![](/assets/cyclegan_t2t_h2z_00.png)
![](/assets/cyclegan_t2t_z2h_00.png)

### Apple to Orange
![](/assets/cyclegan_ok_a2o_00.png)
![](/assets/cyclegan_ok_a2o_01.png)

![](/assets/cyclegan_ng_a2o_00.png)
![](/assets/cyclegan_ng_a2o_01.png)

### Orange to Apple
![](/assets/cyclegan_ok_o2a_00.png)
![](/assets/cyclegan_ok_o2a_01.png)

## Commands:

* **learning-rate**: learning rate, default is 0.0002 (as paper).
* **mode**: 'gx' or 'fy', ignored if is-training is true. 'gx' is for translating image in x domain to y domain. 'fy' is for translating image in y domain to x domain.
* **logs-dir-path**: path to log directory.
* **ckpt-dir-path**: path to checkpoint (work for both training and translation).
* **x-images-dir-path**: path to a directory which contains training images in x domain.
* **y-images-dir-path**: path to a directory which contains training images in y domain.
* **source-image-path**: path to source image for translation.
* **result-image-path**: path to result image for translation.
* **source-images-dir-path**: path to a directory contains source images for translation.
* **result-images-dir-path**: path to a directory contains result images for translation.
* **is-training**: if true, build whole networks and train Cycle-GAN.
* **batch-size**: size of batch (work for both training and translation).
* **history-size**: how many translated images to keep for training the discriminators.
* **checkpoint-step**: save checkpoint every checkpoint-step steps.
* **summary-train-image-step**: log translated images every summary-train-image-step steps.
* **learning-rate-decay-head-at-step**: when to start decay learning rate.
* **learning-rate-decay-tail-at-step**: when to stop training (as paper, learning rate decay to 0.0 in the end of training).


```
# sample command to train Cycle-GAN on Google Cloud Machine Learning Engine

gcloud ml-engine jobs submit training cyclegan_201710032210 \
  --module-name cyclegan.cyclegan \
  --package-path cyclegan \
  --staging-bucket gs://ironhead-mle-staging \
  --region asia-east1 \
  --runtime-version=1.2 \
  --scale-tier=BASIC_GPU \
  -- \
  --is-training \
  --checkpoint-step=20000 \
  --summary-train-image-step=100 \
  --summary-valid-image-step=10000 \
  --batch-size=1 \
  --history-size=50 \
  --logs-dir-path=$(mle_logs_path) \
  --ckpt-dir-path=$(mle_ckpt_path) \
  --x-images-dir-path=$(mle_x_images_dir_path) \
  --y-images-dir-path=$(mle_y_images_dir_path) \
  --learning-rate=0.0002 \
  --learning-rate-decay-head-at-step=133400 \
  --learning-rate-decay-tail-at-step=266800
```

```
# sample command to translate a single image

python -m cyclegan.cyclegan \
  --nois-training \
  --batch-size=1 \
  --ckpt-dir-path=$(local_ckpt_path) \
  --mode=gx \
  --source-image-path=$(local_test_image_path) \
  --result-image-path=./result.jpg
```

```
# sample command to translate all images within a directory (jpg/png)

python -m cyclegan.cyclegan \
  --nois-training \
  --batch-size=16 \
  --ckpt-dir-path=./cyclegan/ckpt/apple2orange/ \
  --mode=gx \
  --source-images-dir-path=/home/ironhead/datasets/cyclegan/apple2orange/testA/ \
  --result-images-dir-path=./cyclegan/test/apple2orange/testA/
```

## Note:

### Train on Google Cloud Machine Learning Engine
Spent 4 days 13 hours to train with horse2zebra dataset on Google Machine Learning Engine (BASIC_GPU).
![Google Machine Learning Engine](/assets/cyclegan_gmle.jpg)

### Failed Pattern

As [vanhuyz](https://github.com/vanhuyz/CycleGAN-TensorFlow) mentioned, "If high contrast background colors between input and generated images are observed (e.g. black becomes white), you should restart your training!". I also had observed that black-white-zebra became white-black-zebra during a failed training case.

![contrast](/assets/cyclegan_failed_00.jpg)
