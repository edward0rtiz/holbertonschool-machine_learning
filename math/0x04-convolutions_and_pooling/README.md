# 0x04 Convolutions and Pooling :robot:

> Convolution is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data. It is a mathematical operation that takes two inputs such as image matrix and a filter or kernel. This porject covers the implemenation of convolutions operations from scratch in python. No tf or keras used just mathematics :)!


At the end of this project I was able to solve these conceptual questions:

* What is a convolution?
* What is max pooling? average pooling?
* What is a kernel/filter?
* What is padding?
* What is “same” padding? “valid” padding?
* What is a stride?
* What are channels?
* How to perform a convolution over an image
* How to perform max/average pooling over an image


## Tasks :heavy_check_mark:

0 Function def convolve_grayscale_valid(images, kernel): that performs a valid convolution on grayscale images:
1. Function def convolve_grayscale_same(images, kernel): that performs a same convolution on grayscale images
2. Function def convolve_grayscale_padding(images, kernel, padding): that performs a convolution on grayscale images with custom padding
3. Function def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on grayscale images
4. Function def convolve_channels(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on images with channels
5. Function def convolve(images, kernels, padding='same', stride=(1, 1)): that performs a convolution on images using multiple kernels
6. Function def pool(images, kernel_shape, stride, mode='max'): that performs pooling on images

## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-convolve_grayscale_valid.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/math/0x04-convolutions_and_pooling/0-convolve_grayscale_valid.py)|
|[1-convolve_grayscale_same.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/math/0x04-convolutions_and_pooling/1-convolve_grayscale_same.py)|
|[2-convolve_grayscale_padding.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/math/0x04-convolutions_and_pooling/2-convolve_grayscale_padding.py)|
| [3-convolve_grayscale.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/math/0x04-convolutions_and_pooling/3-convolve_grayscale.py)|
| [4-convolve_channels.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/math/0x04-convolutions_and_pooling/4-convolve_channels.py)|
| [5-convolve.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/math/0x04-convolutions_and_pooling/5-convolve.py)|
| [6-pool.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/math/0x04-convolutions_and_pooling/6-pool.py)|

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
cd 0x04-convolutions_and_pooling
./main_files/MAINFILE.py
```
