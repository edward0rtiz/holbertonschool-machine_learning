# 0x04 Error Analysis :robot:

> Error Analysis refers to the process of examining dev set examples that your algorithm misclassified, so that we can understand the underlying causes of the errors. This can help us prioritize on which problem deserves attention and how much. It gives us a direction for handling the errors.

At the end of this project I was able to solve these conceptual questions:

* What is the confusion matrix?
* What is type I error? type II?
* What is sensitivity? specificity? precision? recall?
* What is an F1 score?
* What is bias? variance?
* What is irreducible error?
* What is Bayes error?
* How can you approximate Bayes error?
* How to calculate bias and variance
* How to create a confusion matrix

## Tasks :heavy_check_mark:

0. Function def create_confusion_matrix(labels, logits): that creates a confusion matrix
1. Function def sensitivity(confusion): that calculates the sensitivity for each class in a confusion matrix
2. Function def precision(confusion): that calculates the precision for each class in a confusion matrix
3. Function def specificity(confusion): that calculates the specificity for each class in a confusion matrix
4. Function def f1_score(confusion): that calculates the F1 score of a confusion matrix.
5. Lettered answer to the question of how you should approach the following scenarios
      - High Bias, High Variance
      - High Bias, Low Variance
      - Low Bias, High Variance
      - Low Bias, Low Variance
6. Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is

## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-create_confusion.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x04-error_analysis/0-create_confusion.py)|
| [1-sensitivity.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x04-error_analysis/1-sensitivity.py)|
| [2-precision.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x04-error_analysis/2-precision.py)|
| [3-specificity.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x04-error_analysis/3-specificity.py)|
| [4-f1_score.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x04-error_analysis/4-f1_score.py)|
| [5-error_handling](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x04-error_analysis/5-error_handling)|
| [6-compare_and_contrast](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x04-error_analysis/6-compare_and_contrast)|



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
cd 0x04-error_analysis
./main_files/MAINFILE.py
```
