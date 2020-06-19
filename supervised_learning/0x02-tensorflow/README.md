# 0x02 Tensorflow :robot:

> TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. This project covers implementation of DNN using TensorFlow


At the end of this project I was able to solve these conceptual questions:

* What is tensorflow?
* What is a session? graph?
* What are tensors?
* What are variables? constants? placeholders? How do you use them?
* What are operations? How do you use them?
* What are namespaces? How do you use them?
* How to train a neural network in tensorflow
* What is a checkpoint?
* How to save/load a model with tensorflow
* What is the graph collection?
* How to add and get variables from the collection

## Tasks :heavy_check_mark:

0. Function def create_placeholders(nx, classes): that returns two placeholders, x and y, for the neural network
1. Function def create_layer(prev, n, activation) and returns tensor output of the layer
2. Function def forward_prop(x, layer_sizes=[], activations=[]): that creates the forward propagation graph for the neural network
3. Function def calculate_accuracy(y, y_pred): that calculates the accuracy of a prediction
4. Function def calculate_loss(y, y_pred): that calculates the softmax cross-entropy loss of a prediction.
5. Function def create_train_op(loss, alpha): that creates the training operation for the network
6. Function def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"): that builds, trains, and saves a neural network classifier
7. Function def evaluate(X, Y, save_path): that evaluates the output of a neural network

## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-create_placeholders.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/0-create_placeholders.py)|
| [1-create_layer.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/1-create_layer.py)|
| [2-forward_prop.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/2-forward_prop.py)|
| [3-calculate_accuracy.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/3-calculate_accuracy.py)|
| [4-calculate_loss.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/4-calculate_loss.py)|
| [5-create_train_op.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/5-create_train_op.py)|
| [5-create_train_op.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/5-create_train_op.py)|
| [6-train.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/6-train.py)|
| [7-evaluate.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-tensorflow/7-evaluate.py)|



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
cd 0x02-tensorflow
./main_files/MAINFILE.py
```
