# 0x08 Deep Convolutional Architectures :robot:

> Continuing with the CNN's this porject covers the top 5 popular and widely-used deep learning architectures you should know in order to advance your knowledge or deep learning research. In this project the implementation of the disruptives architectures that changed the course rof machine learning are implemented here.

At the end of this project I was able to solve these conceptual questions:

* What is a convolutional layer?
* What is a pooling layer?
* Forward propagation over convolutional and pooling layers
* Back propagation over convolutional and pooling layers
* How to build a CNN using Tensorflow and Keras

## Tasks :heavy_check_mark:

0. Function def inception_block(A_prev, filters): that builds an inception block as described in Going Deeper with Convolutions (2014
1. Function def inception_network(): that builds the inception network as described in Going Deeper with Convolutions (2014)
2. Function def identity_block(A_prev, filters): that builds an identity block as described in Deep Residual Learning for Image Recognition (2015)
3. Function def projection_block(A_prev, filters, s=2): that builds a projection block as described in Deep Residual Learning for Image Recognition (2015)
4. Function def resnet50(): that builds the ResNet-50 architecture as described in Deep Residual Learning for Image Recognition (2015)
5. Function def dense_block(X, nb_filters, growth_rate, layers): that builds a dense block as described in Densely Connected Convolutional Networks
6. Function def transition_layer(X, nb_filters, compression): that builds a transition layer as described in Densely Connected Convolutional Networks
7. Function def densenet121(growth_rate=32, compression=1.0): that builds the DenseNet-121 architecture as described in Densely Connected Convolutional Networks


## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-inception_block.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/0-inception_block.py)|
| [1-inception_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/1-inception_network.py)|
| [2-identity_block.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/2-identity_block.py)|
| [3-projection_block.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/3-projection_block.py)|
| [4-resnet50.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/4-resnet50.py)|
| [5-dense_block.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/5-dense_block.py)|
| [6-transition_layer.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/6-transition_layer.py)|
| [7-densenet121.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/7-densenet121.py)|


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
cd 0x08-deep_cnns
./main_files/MAINFILE.py
```
