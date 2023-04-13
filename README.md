# Complementary Self-Attention Across Approximated DCT Domain towards Improved Class-conditioned Image Generation

Tensorflow implementation for reproducing main results in the paper "Complementary Self-Attention Across Approximated
DCT Domain towards Improved Class-conditioned
Image Generation" by Wuxia Zhang, Xiaoyan Zhang and Jianmin Jiang. This code is based on [Self-Attention GAN](https://github.com/brain-research/self-attention-gan).

[//]: # (<img src="imgs/img1.png"/>)


### Dependencies
python 3.6

TensorFlow 1.5


**Data**

Download Imagenet dataset and preprocess the images into tfrecord files as instructed in [improved gan](https://github.com/openai/improved-gan/blob/master/imagenet/convert_imagenet_to_records.py). Put the tfrecord files into ./data


**Training**

The current batch size is 64x4=256. Larger batch size seems to give better performance. But it might need to find new hyperparameters for G&D learning rate. Note: It usually takes several weeks to train one million steps. 

CUDA_VISIBLE_DEVICES=1,3,5,7 python train_imagenet.py --generator_type test --discriminator_type test --data_dir /raid/zhangwuxia/transform-imagenet/imagenet

**Evaluation**

CUDA_VISIBLE_DEVICES=0 python eval_imagenet.py --generator_type test --data_dir /raid/zhangwuxia/transform/imagenet

[//]: # (### Citing Self-attention GAN)

[//]: # (If you find Self-attention GAN is useful in your research, please consider citing:)

[//]: # ()
[//]: # (```)

[//]: # (@article{Han18,)

[//]: # (  author    = {Han Zhang and)

[//]: # (               Ian J. Goodfellow and)

[//]: # (               Dimitris N. Metaxas and)

[//]: # (               Augustus Odena},)

[//]: # (  title     = {Self-Attention Generative Adversarial Networks},)

[//]: # (  year      = {2018},)

[//]: # (  journal = {arXiv:1805.08318},)

[//]: # (})

[//]: # (```)

**References**
- Self-Attention Generative Adversarial Networks [Paper](https://arxiv.org/abs/1805.08318)
- Spectral Normalization for Generative Adversarial Networks [Paper](https://arxiv.org/abs/1802.05957) 
- cGANs with Projection Discriminator [Paper](https://arxiv.org/abs/1802.05637)
- Non-local Neural Networks [Paper](https://arxiv.org/abs/1711.07971)
