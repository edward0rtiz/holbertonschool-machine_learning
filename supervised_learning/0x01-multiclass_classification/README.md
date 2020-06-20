# 0x01 Multiclass Classification :robot:

> In machine learning, multiclass or multinomial classification is the problem of classifying instances into one of three or more classes (classifying instances into one of two classes is called binary classification).


At the end of this project I was able to solve these conceptual questions:

* What is multiclass classification?
* What is a one-hot vector?
* How to encode/decode one-hot vectors
* What is the softmax function and when do you use it?
* What is cross-entropy loss?
* What is pickling in Python?

## Tasks :heavy_check_mark:

0 Function def one_hot_encode(Y, classes): that converts a numeric label vector into a one-hot matrix
1. Function def one_hot_decode(one_hot): that converts a one-hot matrix into a vector of labels
2. Updated the class DeepNeuralNetwork (based on 23-deep_neural_network.py) Created the instance method def save(self, filename)
3. Updated the class DeepNeuralNetwork to perform multiclass classification (based on 2-deep_neural_network.py)
4. Updated the class DeepNeuralNetwork to allow different activation functions (based on 3-deep_neural_network.py) Updated the __init__ method to def __init__(self, nx, layers, activation='sig')


## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-one_hot_encode.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-multiclass_classification/0-one_hot_encode.py)|
| [1-one_hot_decode.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-multiclass_classification/1-one_hot_decode.py)|
| [2-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-multiclass_classification/2-deep_neural_network.py)|
| [3-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-multiclass_classification/3-deep_neural_network.py)|
| [4-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-multiclass_classification/4-deep_neural_network.py)|


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
cd 0x01-multiclass_classification
./main_files/MAINFILE.py
```
