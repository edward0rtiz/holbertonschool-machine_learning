# 0x05 Regularization :robot:

> Regularization. This is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. This project covers regularization algorithms in machine learning

At the end of this project I was able to solve these conceptual questions:

* What is regularization? What is its purpose?
* What is are L1 and L2 regularization? What is the difference between the two methods?
* What is dropout?
* What is early stopping?
* What is data augmentation?
* How do you implement the above regularization methods in Numpy? Tensorflow?
* What are the pros and cons of the above regularization methods?

## Tasks :heavy_check_mark:

0. Function def l2_reg_cost(cost, lambtha, weights, L, m): that calculates the cost of a neural network with L2 regularization
1. Function def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L): that updates the weights and biases of a neural network using gradient descent with L2 regularization
2. Function def l2_reg_cost(cost): that calculates the cost of a neural network with L2 regularization
3. Function def l2_reg_create_layer(prev, n, activation, lambtha): that creates a tensorflow layer that includes L2 regularization
4. Function def dropout_forward_prop(X, weights, L, keep_prob): that conducts forward propagation using Dropout.
5. Function def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L): that updates the weights of a neural network with Dropout regularization using gradient descent
6. Function def dropout_create_layer(prev, n, activation, keep_prob): that creates a layer of a neural network using dropout
7. Function def early_stopping(cost, opt_cost, threshold, patience, count): that determines if you should stop gradient descent early

## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-l2_reg_cost.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/0-l2_reg_cost.py)|
| [1-l2_reg_gradient_descent.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/1-l2_reg_gradient_descent.py)|
| [2-l2_reg_cost.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/2-l2_reg_cost.py)|
| [3-l2_reg_create_layer.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/3-l2_reg_create_layer.py)|
| [4-dropout_forward_prop.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/4-dropout_forward_prop.py)|
| [5-dropout_gradient_descent.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/5-dropout_gradient_descent.py)|
| [6-dropout_create_layer.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/6-dropout_create_layer.py)|
| [7-early_stopping.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x05-regularization/7-early_stopping.py)|



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
cd 0x05-regularization
./main_files/MAINFILE.py
```
