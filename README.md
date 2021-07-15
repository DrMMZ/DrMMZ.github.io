**[RetinaNet for Object Detection](https://github.com/DrMMZ/RetinaNet)**

[RetinaNet](https://arxiv.org/abs/1708.02002) is an efficient one-stage object detector trained with the focal loss. This [repository](https://github.com/DrMMZ/RetinaNet) is a TensorFlow2 implementation of RetinaNet and its applications, aiming for creating a tool in object detection task that can be easily extended to other datasets or used in building projects.

The following are example real-time detections using RetinaNet, randomly selected from un-trained images.

1. My own dataset, *empty returns operations (ERO)*, is a collection of images such that each contains empty beer, wine and liquor cans or bottles in densely packed scenes that can be returned for refunds. The goal is to count the number of returns fast and accurately, instead of manually checking by human (specially for some people like me who is bad on counting). The dataset (as of July 15 2021) consists of 47 labeled cellphone images in cans, variety of positions. If you are interested in contributing to this dataset, please [email](mailto:mmzhangist@gmail.com) me. 
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/ero_movie.gif" width='360' height='360'/>
</p> 

2. The [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) dataset, focusing on detection in densely packed scenes. Indeed, our ERO detection above used transfer learning from it.
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/sku_movie.gif" width='360' height='360'/>
</p>

3. The [nuclei](https://www.kaggle.com/c/data-science-bowl-2018) dataset, identifying the cells’ nuclei. 
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/nuclei_movie.gif" width='360' height='360'/>
</p> 

----

**[Ensemble Model: ResNet + FPN](https://github.com/DrMMZ/ResFPN)**

This is an implementation of [*ResFPN*](https://github.com/DrMMZ/ResFPN) on Python 3 and TensorFlow 2. The model classifies images by ensembling predictions from [Residual Network](https://arxiv.org/abs/1512.03385) (ResNet) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) (FPN), and can be trained by minimizing [focal loss](https://arxiv.org/abs/1708.02002). 

Below are example classifications using ResFPN on the [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset randomly selected from un-trained images.

<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/flower_movie.gif" width='480' height='360'/>
</p>
