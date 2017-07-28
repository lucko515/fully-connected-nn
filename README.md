# Fully Connected Neural Network - Numpy, Tensorflow and Keras

The Kaggles mushrooms dataset classified with fully connected neural networks. The dataset is pretty simple and we can easily achieve 100% accuracy with most of models.

The bonus code in this repository is implementation of feed forward netowrk using Keras and Tensorflow library. That is for people interested in learning those two libraries.
## Dataset

The dataset used in this project can be found on Kaggle [here](https://www.kaggle.com/uciml/mushroom-classification).

## Install

### &nbsp;&nbsp;&nbsp; Supported Python version
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Python version used in this project: 3.5+

### &nbsp;&nbsp;&nbsp; Libraries used

> *  [Pandas](http://pandas.pydata.org) 0.18.0
> *  [Numpy](http://www.numpy.org) 1.10.4
> *  [Matplotlib](https://matplotlib.org) 1.5.1
> *  [Tensorflow](http://tensorflow.org) 1.2.0
> *  [Keras](https://keras.io)

## Code

The numpy testing file is **Numpy NN.ipynb** - use this to test Numpy version of feed forward network on this dataset.

The tensorflow version of network can be found in **Tensorflow NN.ipynb**.

The Keras version of the network can be found in **Keras NN.ipynb**.

Methods used to preprocess this dataset are inside **data_handlers.py**.

There is some bonus code inside folder *activation_weights_init_losses*.
Inside this folder you can find implementation of many activation functions used in DL, different techniques for weight initialization and two loss functions (softmax loss and mean_squared_error).

## Run

To run this project you will need some software, like Anaconda, which provides support for running .ipynb files (Jupyter Notebook).

After making sure you have that, you can run from a terminal or cmd next lines:

[for example: ]

`ipython notebook 'Numpy NN.ipynb'`

or

`jupyter notebook 'Numpy NN.ipynb'`

If a file has an extension **.py** instead of **.ipynb** - you can start it as well from your terminal/cmd with this line:

`python name_of_a_file.py`


## License

MIT License

Copyright (c) 2017 Luka Anicin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
