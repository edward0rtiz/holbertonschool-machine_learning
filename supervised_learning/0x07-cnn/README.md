# 0x07 Convolutional Neural Networks :robot:

> In deep learning, a convolutional neural network is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks, based on their shared-weights architecture and translation invariance characteristics. This project covers the implementation of CNN's

At the end of this project I was able to solve these conceptual questions:

* What is a convolutional layer?
* What is a pooling layer?
* Forward propagation over convolutional and pooling layers
* Back propagation over convolutional and pooling layers
* How to build a CNN using Tensorflow and Keras

## Tasks :heavy_check_mark:

0. Function def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)): that performs forward propagation over a convolutional layer of a neural network
1. Function def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs forward propagation over a pooling layer of a neural network
2. Function def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)): that performs back propagation over a convolutional layer of a neural network
3. Function def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs back propagation over a pooling layer of a neural network
4. Function def lenet5(x, y): that builds a modified version of the LeNet-5 architecture using tensorflow
5. Function def lenet5(X): that builds a modified version of the LeNet-5 architecture using keras


## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-conv_forward.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/0-conv_forward.py)|
| [1-pool_forward.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/1-pool_forward.py)|
| [2-conv_backward.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/2-conv_backward.py)|
| [3-pool_backward.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/3-pool_backward.py)|
| [4-lenet5.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/4-lenet5.py)|
| [5-lenet5.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/5-lenet5.py)|


## Additional info :construction:
### Resources

- Python 3.5 / 3.8
- Pycharm Professional 2020.1
- absl-py 0.9.0
- astor 0.8.1
- cycler 0.10.0
- gast 0.3.3
- grpcio 1.29.0
- h5py 2.10.0
- importlib-metadata 1.6.0
- Keras-Applications 1.0.8
- Keras-Preprocessing 1.1.2
- kiwisolver 1.1.0
- Markdown 3.2.2
- matplotlib 3.0.3
- numpy 1.18.4
- Pillow 7.1.2
- protobuf 3.11.3
- pycodestyle 2.5.0
- pyparsing 2.4.7
- python-dateutil 2.8.1
- scipy 1.4.1
- six 1.14.0
- tensorboard 1.12.2
- termcolor 1.1.0
- Werkzeug 1.0.1
- zipp 1.2.0


### Try It On Your Machine :computer:
```bash
git clone https://github.com/edward0rtiz/holbertonschool-machine_learning.git
cd 0x07-cnn
./main_files/MAINFILE.py
```
