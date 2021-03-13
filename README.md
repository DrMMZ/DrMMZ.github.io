**[Ensemble Model: ResNet + FPN](https://github.com/DrMMZ/ResFPN)**

This is an implementation of [*ResFPN*](https://github.com/DrMMZ/ResFPN) on Python 3 and TensorFlow 2. The model classifies images by ensembling predictions from [Residual Network](https://arxiv.org/abs/1512.03385) (ResNet) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) (FPN). 

The repository includes:
* source code of ResFPN built on ResNet50/101 and FPN, shown in the [model](https://github.com/DrMMZ/ResFPN/tree/main/model) folder; and
* jupyter notebook demonstration the use of ResFPN in training, evaluation and visualization, shown in the [tutorial](https://github.com/DrMMZ/ResFPN/tree/main/tutorial) folder.

----

**[ProtoNets-TensorFlow](https://github.com/DrMMZ/ProtoNets-TensorFlow)**

Implement [ProtoNets for few-shot learning](https://arxiv.org/abs/1703.05175) in TensorFlow 2, and perform experiments on the [COVIDx dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md).

See [here](https://github.com/DrMMZ/ProtoNets-TensorFlow/blob/master/ProtoNets/ProtoNets.py) for the implementation of `ProtoNets` and [here](https://github.com/DrMMZ/ProtoNets-TensorFlow/blob/master/Experiments/COVIDx.ipynb) for the notebook demonstration on the COVIDx dataset experiments.

----

**[NumPy Implementation of Fully Connected Neural Networks](https://github.com/DrMMZ/drmmz.github.io/blob/master/NN_numpy.ipynb)**

Implemented a vectorized fully connected L-layer ReLU network for any integer L>0 with L1 or L2-regularized softmax loss and stochastic gradient descent in Numpy.

The implementation is based on *Derivatives in fully connected neural networks* by Ming Ming Zhang. In addition, numeric stability and gradients checking are added.
