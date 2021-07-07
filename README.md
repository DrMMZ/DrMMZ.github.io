**[RetinaNet for Object Detection](https://github.com/DrMMZ/RetinaNet)**

[RetinaNet](https://arxiv.org/abs/1708.02002) is an efficient one-stage object detector trained with the focal loss. This repository is a TensorFlow2 implementation of RetinaNet and its applications, aiming for creating a tool in object detection task that can be easily extended to other datasets or used in building projects.

Below are example detections on the [nuclei](https://www.kaggle.com/c/data-science-bowl-2018) dataset randomly selected from un-trained images.

<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/nuclei_movie.gif" width='360' height='360'/>
</p> 

----

**[Ensemble Model: ResNet + FPN](https://github.com/DrMMZ/ResFPN)**

This is an implementation of [*ResFPN*](https://github.com/DrMMZ/ResFPN) on Python 3 and TensorFlow 2. The model classifies images by ensembling predictions from [Residual Network](https://arxiv.org/abs/1512.03385) (ResNet) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) (FPN), and can be trained by minimizing [focal loss](https://arxiv.org/abs/1708.02002). 

Below are example classifications using ResFPN on the [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset.

<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/flower_movie.gif" width='480' height='360'/>
</p>

----

**[ProtoNets-TensorFlow](https://github.com/DrMMZ/ProtoNets-TensorFlow)**

Implement [ProtoNets for few-shot learning](https://arxiv.org/abs/1703.05175) in TensorFlow 2, and perform experiments on the [COVIDx dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md).

See [here](https://github.com/DrMMZ/ProtoNets-TensorFlow/blob/master/ProtoNets/ProtoNets.py) for the implementation of `ProtoNets` and [here](https://github.com/DrMMZ/ProtoNets-TensorFlow/blob/master/Experiments/COVIDx.ipynb) for the notebook demonstration on the COVIDx dataset experiments.
