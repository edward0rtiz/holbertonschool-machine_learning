# 0x03 Optimization :robot:

> Optimization is the most essential ingredient in the recipe of machine learning algorithms. It starts with defining some kind of loss function/cost function and ends with minimizing the it using one or the other optimization routine. This project covers the most important optimization algorithms for implement supervised learning models


At the end of this project I was able to solve these conceptual questions:

* What is a hyperparameter?
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

## Tasks :heavy_check_mark:

0. Function def normalization_constants(X): that calculates the normalization (standardization) constants of a matrix
1. Function def normalize(X, m, s): that normalizes (standardizes) a matrix
2. Function def shuffle_data(X, Y): that shuffles the data points in two matrices the same way
3. Function def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"): that trains a loaded neural network model using mini-batch gradient descent
4. Function def moving_average(data, beta): that calculates the weighted moving average of a data set.
5. Function def update_variables_momentum(alpha, beta1, var, grad, v): that updates a variable using the gradient descent with momentum optimization algorithm
6. Function def create_momentum_op(loss, alpha, beta1): that creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm
7. Function def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s): that updates a variable using the RMSProp optimization algorithm
8. Function def create_RMSProp_op(loss, alpha, beta2, epsilon): that creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm
9. Function def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t): that updates a variable in place using the Adam optimization algorithm
10. Function def create_Adam_op(loss, alpha, beta1, beta2, epsilon): that creates the training operation for a neural network in tensorflow using the Adam optimization algorithm
11. Function def learning_rate_decay(alpha, decay_rate, global_step, decay_step): that updates the learning rate using inverse time decay in numpy
12. Function def learning_rate_decay(alpha, decay_rate, global_step, decay_step): that creates a learning rate decay operation in tensorflow using inverse time decay
13. Function def batch_norm(Z, gamma, beta, epsilon): that normalizes an unactivated output of a neural network using batch normalization
14. Function def create_batch_norm_layer(prev, n, activation): that creates a batch normalization layer for a neural network in tensorflow
15. Function def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'): that builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization

## Results :chart_with_upwards_trend:

| Filename ||||
| ------ |---|----|----|
| [0-norm_constants.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/0-norm_constants.py)| [1-normalize.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/1-normalize.py)|[2-shuffle_data.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/2-shuffle_data.py)|[3-mini_batch.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/3-mini_batch.py)|
|[4-moving_average.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/4-moving_average.py)|[5-momentum.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/5-momentum.py)| [6-momentum.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/6-momentum.py)|[7-RMSProp.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/7-RMSProp.py)|
|[8-RMSProp.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/8-RMSProp.py)|[9-Adam.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/9-Adam.py)|[10-Adam.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/10-Adam.py)|[11-learning_rate_decay.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/11-learning_rate_decay.py)|
|[12-learning_rate_decay.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/12-learning_rate_decay.py)|[13-batch_norm.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/13-batch_norm.py)|[14-batch_norm.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/14-batch_norm.py)|[15-model.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-optimization/15-model.py)


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
cd 0x03-optimization
./main_files/MAINFILE.py
```
