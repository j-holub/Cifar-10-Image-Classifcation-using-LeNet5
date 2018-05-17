# Cifar 10 using LeNet5 and Keras

This repository has **4** different tasks performed with **LeNet5** on the **Cifar10** Dataset

* Cifar 10 Image Classification
* Cifar  9 Image Classification (dropped the last class)
* Cifar 9 where *Truck* and *Automobile* are treated as the same class
* Cifar 9 where *Truck* and *Bird* are treated as the same class

## Network

![LeNet5 Image](https://i.stack.imgur.com/tLKYz.png)

LeNet5 was introduced by [Yan LeCun](http://yann.lecun.com) back in 1998. It's a form of a convolutional network originally intented for the classification of the MNIST Dataset (Handwritten Digits).

Here I made a few changes to the network:

* *Activation Functions*: ReLu
* *Kernel Size*: 3x3

## Training

Training went on for **10** epochs with a **batch size** of 32

* *Loss Function*: Categorical Cross Entropy
* *Optimizer*: SGD

## Results

| Task                        | Loss  | Accuracy |
|-----------------------------|-------|----------|
| Cifar 10                    | 3.589 | 0.0978   |
| Cifar 9                     | 2.171 | 0.133    |
| Cifar 9 Truck as Automobile | 2.197 | 0.1      |
| Cifar 9 Truck as Bird       | 2.163 | 0.2      |

#### Cifar 9

It makes sense, that the **loss** went down when the network has to classify one class less because it can focus the weight adaptation in a better way for these 9 classes compared to when it has another class to classify.

But at the same time the **accuracy** went down which I find odd to be honest.

#### Cifar 9 with Truck as Automobile

The **loss** went slightly up when I labeled *Truck* as *automobile*. Naturally Trucks are a kind of automobile so I would assume they have a lot of visual features in common (at least to me as a human), such as tires and windows for example.

The **accuracy** went down again on this one which kind of makes sense since we have more training examples for one specific class. Given that *Trucks* and *Autombiles* have a lot of visual features in common from the networks perspective, we tend to overfit on that class. At the same time we make it more ambigious because we add a wider variaty of features to that class.

#### Cifar 9 with Truck as Bird

Now *Trucks* and *Birds* should really have nothing in common, but surprisingly the **Loss** went down and the **Accuracy** went up. To be honest, I have no explanation to this.

## Running

The implementation was done using Keras with TensorFlow as the backed. Everything you need will be installed via the *requirements.txt* file

```
pip3 install -r requirements.txt
```

You should preferably use a virtual environment.

The **Cifar10** dataset will be downloaded when you first run one of the python scripts

## References

* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org)
* [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [LeNet5](http://yann.lecun.com/exdb/lenet/)
* [Yan LeCun](http://yann.lecun.com)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
