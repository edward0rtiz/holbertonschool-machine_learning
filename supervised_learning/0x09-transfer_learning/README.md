# 0x09 Transfer Learning :robot:

> Transfer Learning is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. In this project there will be some experiments with CIFAR-10 and implementing Transfer Learning concept 

At the end of this project I was able to solve these conceptual questions:

* What is a transfer learning?
* What is fine-tuning?
* What is a frozen layer? How and why do you freeze a layer?
* How to use transfer learning with Keras applications

## Tasks :heavy_check_mark:


0. Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:
   - You must use one of the applications listed in Keras Applications
   - Your script must save your trained model in the current working directory as cifar10.h5
    - Your saved model should be compiled
    - Your saved model should have a validation accuracy of 87% or higher
    - Your script should not run when the file is imported

## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-transfer.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x09-transfer_learning/0-transfer.py)|
| [0-transfer_fine_tuning.ipynb.zip](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x09-transfer_learning/0-transfer_fine_tuning.ipynb.zip)|

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
cd 0x09-transfer_learning
./0-main.py
```
