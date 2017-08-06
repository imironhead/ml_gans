# Replicated "Image-to-Image Translation with Conditional Adversarial Networks"


### Commands:

* **lambda-value**: weight of L1 loss for the generator.
* **learning-rate**: learning rate.
* **logs-dir-path**: path to the directory for logs.
* **ckpt-dir-path**: path to the directory for checkpoint.
* **images-path**: path to the source-target image for training.
* **source-image-path**: path to the source image for translation.
* **target-image-path**: path to the target image for translation.
* **is-training**: build and train the model.
* **nois-training**
* **swap-images**: swap source-target image to target-source one.
* **noswap-images**
* **batch-size**: batch size for training.
* **crop-image-size**: how to crop the training image for training.

### Results:

* **lambda-value**: 200.0
* **learning-rate**: 0.001

## Training

![facades](/assets/pix2pix_training_00.png)

![facades](/assets/pix2pix_training_01.png)

## Test

![facades 1](/assets/pix2pix_test_1.png)

![facades 1](/assets/pix2pix_test_10.png)

![facades 1](/assets/pix2pix_test_32.png)

### Dataset:

Found [here](https://github.com/junyanz/CycleGAN)!

## facades

```
@INPROCEEDINGS{Tylecek13,
  author = {Radim Tyle{\v c}ek, Radim {\v S}{\' a}ra},
  title = {Spatial Pattern Templates for Recognition of Objects with Regular Structure},
  booktitle = {Proc. GCPR},
  year = {2013},
  address = {Saarbrucken, Germany},
}
```
