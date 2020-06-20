# 0x06 Keras :robot:

> Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible.

At the end of this project I was able to solve these conceptual questions:

* What is Keras?
* What is a model?
* How to instantiate a model (2 ways)
* How to build a layer
* How to add regularization to a layer
* How to add dropout to a layer
* How to add batch normalization
* How to compile a model
* How to optimize a model
* How to fit a model
* How to use validation data
* How to perform early stopping
* How to measure accuracy
* How to evaluate a model
* How to make a prediction with a model
* How to access the weights/outputs of a model
* What is HDF5?
* How to save and load a model’s weights, a model’s configuration, and the entire model

## Tasks :heavy_check_mark:

0. Function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library
1. Function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library
2. Function def optimize_model(network, alpha, beta1, beta2): that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics
3. Function def one_hot(labels, classes=None): that converts a label vector into a one-hot matrix
4. Function def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False): that trains a model using mini-batch gradient descen
5. Based on 4-train.py, updated the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False): to also analyze validaiton data
6. Based on 5-train.py, updated the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False): to also train the model using early stopping
7. Based on 6-train.py, updated the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False): to also train the model with learning rate decay
8. Based on 7-train.py, updated the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model
9. Functions def save_model(network, filename): saves an entire model and def load_model(filename): loads an entire model
10. Functions def save_weights(network, filename, save_format='h5'): saves a model’s weights and def load_weights(network, filename): loads a model’s weights:
11. Functions def save_config(network, filename): saves a model’s configuration in JSON format and def load_config(filename): loads a model with a specific configuration
12. Function def test_model(network, data, labels, verbose=True): that tests a neural network


## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-sequential.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/0-sequential.py)|
| [1-input.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/1-input.py)|
| [2-optimize.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/2-optimize.py)|
| [3-one_hot.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/3-one_hot.py)|
| [4-train.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/4-train.py)|
| [5-train.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/5-train.py)|
| [6-train.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/6-train.py)|
| [7-train.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/7-train.py)|
| [8-train.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/8-train.py)|
| [9-model.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/9-model.py)|
| [10-weights.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/10-weights.py)|
| [11-config.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/11-config.py)|
| [12-test.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/12-test.py)|
| [13-predict.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x06-keras/13-predict.py)|



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
cd 0x06-keras
./main_files/MAINFILE.py
```
