# 0x00 Binary Classification :robot:

> Binary Classification is a problem studied in machine learning. It is a type of supervised learning, a method of machine learning where the categories are predefined, and is used to categorize new probabilistic observations into said categories.


At the end of this project I was able to solve these conceptual questions:

* What is a model?
* What is supervised learning?
* What is a prediction?
* What is a node?
* What is a weight?
* What is a bias?
* What are activation functions? Sigmoid? Tanh? Relu? Softmax?
* What is a layer?
* What is a hidden layer?
* What is Logistic Regression?
* What is a loss function?
* What is a cost function?
* What is forward propagation?
* What is Gradient Descent?
* What is back propagation?
* What is a Computation Graph?
* How to initialize weights/biases
* The importance of vectorization
* How to split up your data

## Tasks :heavy_check_mark:

0 Class Neuron that defines a single neuron performing binary classification
1. Class Neuron that defines a single neuron performing binary classification (Based on 0-neuron.py)
2. Class Neuron that defines a single neuron performing binary classification (Based on 1-neuron.py) Includes public method def forward_prop(self, X)
3. Class Neuron that defines a single neuron performing binary classification (Based on 2-neuron.py) Includes public method def cost(self, Y, A)
4. Class Neuron that defines a single neuron performing binary classification (Based on 3-neuron.py) Includes public method def evaluate(self, X, Y)
5. Class Neuron that defines a single neuron performing binary classification (Based on 4-neuron.py) Includes public method def gradient_descent(self, X, Y, A, alpha=0.05)
6. Class Neuron that defines a single neuron performing binary classification (Based on 5-neuron.py) Includes public method def train(self, X, Y, iterations=5000, alpha=0.05)
7. Class Neuron that defines a single neuron performing binary classification (Based on 6-neuron.py) Includes public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)
8. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification
9. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 8-neural_network.py) Includes Weights, biases, and activations as private instances
10. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 9-neural_network.py) Includes public method def forward_prop(self, X)
11. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 10-neural_network.py) Includes public method def cost(self, Y, A)
12. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 11-neural_network.py) Includes public method def evaluate(self, X, Y)
13. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 12-neural_network.py) Includes public method def gradient_descent(self, X, Y, A1, A2, alpha=0.05)
14. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 13-neural_network.py) Includes public method def train(self, X, Y, iterations=5000, alpha=0.05)
15. Class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 14-neural_network.py) Updated public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)
16. Class DeepNeuralNetwork that defines a deep neural network performing binary classification
17. Class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 16-deep_neural_network.py) Includes Weights, biases, activations, layers and cache as private instances.
18. Class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 17-deep_neural_network.py) Includes public method def forward_prop(self, X)
19. Class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 18-deep_neural_network.py) Includes public method def cost(self, Y, A)
20. Class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 19-deep_neural_network.py) Includes public method def evaluate(self, X, Y)
21. Class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 20-deep_neural_network.py) Includes public method def gradient_descent(self, Y, cache, alpha=0.05)
22. Class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 21-deep_neural_network.py) Includes public method def train(self, X, Y, iterations=5000, alpha=0.05)
23. Class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 22-deep_neural_network.py) Updated public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)

## Results :chart_with_upwards_trend:

| Filename ||||||
| ------ |---|----|----|----|---|
| [0-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/0-neuron.py)|[1-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/1-neuron.py)|[2-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/2-neuron.py)|[3-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/3-neuron.py)|[4-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/4-neuron.py)|[5-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/5-neuron.py)|
| [6-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/6-neuron.py)|[7-neuron.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/7-neuron.py)|[8-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/8-neural_network.py)|[9-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/9-neural_network.py)|[10-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/10-neural_network.py)|[11-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/11-neural_network.py)|
| [12-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/12-neural_network.py)|[13-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/13-neural_network.py)|[14-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/14-neural_network.py)|[15-neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/15-neural_network.py)|[16-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/16-deep_neural_network.py)|[17-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/17-deep_neural_network.py)|
| [18-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/18-deep_neural_network.py)|[19-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/19-deep_neural_network.py)|[20-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/20-deep_neural_network.py)|[21-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/21-deep_neural_network.py)|[22-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/22-deep_neural_network.py)|[23-deep_neural_network.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-binary_classification/23-deep_neural_network.py)|

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
cd 0x00-binary_classification
./main_files/MAINFILE.py
```
