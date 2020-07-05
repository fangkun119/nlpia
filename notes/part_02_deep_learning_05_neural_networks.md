# Part 2: Deep Learning

> `Part 1` is about statistics-driven vector space models, but consider only linear relationships between words, which make developers often had to use human judgment to design feature extractors and select model parameters

> `Part 2` accomplish most of the tedious feature extraction and often more accurate than hand-tuned feature extrations 


# Ch 05. Baby steps with neural networks (perceptrons and backpropagation)

> this chapter does not introduction, it introduce `neural networks` in high level 

> for more reference, below books are good resource 
> 
> * Deep Learning with Python, by François Chollet (Manning, 2017), is a deep dive into the wonders of deep learning by the creator of Keras himself.
> * Grokking Deep Learning, by Andrew Trask (Manning, 2017), is a broad overview of deep learning models and practices.


## 5.1. Neural networks, the ingredient list

neuron cell: 

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig01_alt.jpg)


### 5.1.1. Perceptron

### 5.1.2. A numerical perceptron

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig02_alt.jpg)

### 5.1.3. Detour through bias
### 5.1.4. Let’s go skiing—the error surface

<b>perceptron structure and bias (`X`<sub>`b`</sub> `= 1`)</b>

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig03_alt.jpg)

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0159-01.jpg)

~~~python
>>> import numpy as np
 
>>> example_input = [1, .2, .1, .05, .2]
>>> example_weights = [.2, .12, .4, .6, .90]
 
>>> input_vector = np.array(example_input)
>>> weights = np.array(example_weights)
>>> bias_weight = .2
 
>>> activation_level = np.dot(input_vector, weights) +\
...     (bias_weight * 1)
>>> activation_level
0.674

>>> threshold = 0.5
>>> if activation_level >= threshold:
...    perceptron_output = 1
... else:
...    perceptron_output = 0
>>> perceptron_output)
1
~~~

<b>how a single perceptron is trained</b>

perceptron learns by altering the weights up or down as a function of how wrong the system’s guess was for a given input

> * weights of an untrained neuron start out random, and usually chosen from a normal distribution
> 	* starting the weights (including the bias weight) at zero would lead only to an output of zero
> * for each example the weights are readjusted a small amount based on whether the neuron output was what you wanted or not
> * with enough examples (and under the right conditions), the error should tend toward zero

<b>how to adjust the weight of a single layer perceptron</b>

> each weight is adjusted by how much it contributed to the resulting error

~~~python
>>> expected_output = 0
>>> new_weights = []
>>> for i, x in enumerate(example_input):
...     new_weights.append(weights[i] + (expected_output -\
...         perceptron_output) * x)
 >>> weights = np.array(new_weights)
 
>>> example_weights
[0.2, 0.12, 0.4, 0.6, 0.9]
>>> weights
[-0.8  -0.08  0.3   0.55  0.7]
~~~

> a complete toy example of basic (single layer) perceptron 

~~~python
>>> # data
>>> sample_data = [[0, 0],  # False, False
...                [0, 1],  # False, True
...                [1, 0],  # True, False
...                [1, 1]]  # True, True
 
>>> expected_results = [0,  # (False OR False) gives False
...                     1,  # (False OR True ) gives True
...                     1,  # (True  OR False) gives True
...                     1]  # (True  OR True ) gives True
 
>>> activation_threshold = 0.5

>>> # initialize the weights
>>> from random import random
>>> import numpy as np
>>> weights = np.random.random(2)/1000  # Small random float 0 < w < .001
>>> weights
[5.62332144e-04 7.69468028e-05]
>>> bias_weight = np.random.random() / 1000
>>> bias_weight
0.0009984699077277136

>>> # training to update the weights
>>> for iteration_num in range(5):
...     correct_answers = 0
...     for idx, sample in enumerate(sample_data):
...         input_vector = np.array(sample)
...         weights = np.array(weights)
...         activation_level = np.dot(input_vector, weights) +\
...             (bias_weight * 1)
...         if activation_level > activation_threshold:
...             perceptron_output = 1
...         else:
...             perceptron_output = 0
...         if perceptron_output == expected_results[idx]:
...             correct_answers += 1
...         new_weights = []
...         for i, x in enumerate(sample):
...             new_weights.append(weights[i] + (expected_results[idx] -\
...                 perceptron_output) * x)
...         bias_weight = bias_weight + ((expected_results[idx] -\
...             perceptron_output) * 1)
...         weights = np.array(new_weights)
...     print('{} correct answers out of 4, for iteration {}'\
...         .format(correct_answers, iteration_num))
3 correct answers out of 4, for iteration 0
2 correct answers out of 4, for iteration 1
3 correct answers out of 4, for iteration 2
4 correct answers out of 4, for iteration 3
4 correct answers out of 4, for iteration 4
~~~

> linear seperable space 

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig04_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig04_alt.jpg) 

> non-linear seperable space, and the `basic (single layer) perceptron` (linear model) won’t convergent

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig05_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig05_alt.jpg)

<b>how the weights of a multiple-layers neural network are trained</b>

> error between truth and prediction

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0167-01.jpg) 

> cost function to minimize

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0168-01.jpg)

> multiple layer neural networks

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig06_alt.jpg)

> need a way to calculate the amount a particular weight (w1i in figure 5.6) contributed to the error given that it contributed to the error via other weights (w<sub>1j</sub>) and (w<sub>2j</sub>) in the next layer. And this way is `backpropagation`

<b>backpropagation (</b>short for backpropagation of the errors<b>)</b> 

> * <b>target</b>: given the input, the output, and the expected value, discover the appropriate amount to update a specific weight
> * <b>step 1</b>: change the perceptron’s activation function to a new one which is `non-linear` and `continuously differentiable` (more smooth than a differentiable function, can canculate `derivative` and `partial derivatives`, like `sigmoid` function,  1 / ( 1 - e<sup>-x</sup>)
> * <b>step 2</b>: calculate MSE
>
> 	![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0171-01a.jpg)
> * <b>step 3</b>: </br> 
> 	for single-layer network, the cost function is minimized like below  
>	![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig08_alt.jpg)
>	for multi-layer network, need to tweak the error function, i.e., something that represents the aggregate error across all inputs for a given set of weights</br>

### 5.1.5. Off the chair lift, onto the slope

> local minima problem in gradient descent

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/05fig09_alt.jpg)

### 5.1.6. Let’s shake things up a bit

batch learning

> aggregating the error for all the training examples and skiing down the slope as best you could

stochastic gradient descent

> update the weights after each training example, rather than after looking at all the training examples
> 
> * in practice stochastic gradient descent proves quite effective in avoiding local minima in most cases
> * the downfall of this approach is that it’s slow (calculating the forward pass and backpropagation, and then updating the weights after each example, adds that much time) 

mini-batch

> a small subset of the training set is passed in and the associated errors are aggregated as in full batch

<b>backpropagation</b>

[https://en.wikipedia.org/wiki/Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

### 5.1.7. Keras: Neural networks in Python

> `Keras`: a high-level wrapper with an accessible API for Python, the exposed API can be used with three different backends almost interchangeably
> 
> * Theano
> * TensorFlow from Google
> * CNTK from Microsoft

example: 

~~~python
>>> import numpy as np
>>> from keras.models import Sequential
>>> from keras.layers import Dense, Activation
>>> from keras.optimizers import SGD
>>> # Our examples for an exclusive OR.
>>> x_train = np.array([[0, 0],
...                     [0, 1],
...                     [1, 0],
...                     [1, 1]])
>>> y_train = np.array([[0],
...                     [1],
...                     [1],
...                     [0]])
>>> model = Sequential()
>>> num_neurons = 10
>>> model.add(Dense(num_neurons, input_dim=2))
>>> model.add(Activation('tanh'))
>>> model.add(Dense(1))
>>> model.add(Activation('sigmoid'))
>>> model.summary()
>>> #10 neurons, each with two weights (one for each value in the input vector), and one weight for the bias gives you 30 weights to learn
Layer (type)                 Output Shape              Param #
=================================================================
dense_18 (Dense)             (None, 10)                30
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0
_________________________________________________________________
dense_19 (Dense)             (None, 1)                 11
_________________________________________________________________
activation_7 (Activation)    (None, 1)                 0
=================================================================
Total params: 41.0
Trainable params: 41.0
Non-trainable params: 0.0

>>> # sgd: stochastic gradient descent 
>>> # lr: learning rate
>>> # loss: loss function
>>> # compile: build the model but doesn’t yet train the model
>>> sgd = SGD(lr=0.1)
>>> model.compile(loss='binary_crossentropy', optimizer=sgd,
...     metrics=['accuracy'])

>>> # the weights are initialized, and you can use this random state to try to predict from your dataset, but you’ll only get random guesses
>>> model.predict(x_train)
[[ 0.5       ]
 [ 0.43494844]
 [ 0.50295198]
 [ 0.42517585]]
 
>>> # train model
>>> model.fit(x_train, y_train, epochs=100)
Epoch 1/100
4/4 [==============================] - 0s - loss: 0.6917 - acc: 0.7500
Epoch 2/100
4/4 [==============================] - 0s - loss: 0.6911 - acc: 0.5000
Epoch 3/100
4/4 [==============================] - 0s - loss: 0.6906 - acc: 0.5000
...
Epoch 100/100
4/4 [==============================] - 0s - loss: 0.6661 - acc: 1.0000

>>> #as it looked at what was a tiny dataset over and over, it finally figured out what was going on. It “learned” what exclusive-or (XOR) was 
>>> model.predict_classes(x_train))
4/4 [==============================] - 0s
[[0]
 [1]
 [1]
 [0]]
>>> model.predict(x_train))
4/4 [==============================] - 0s
[[ 0.0035659 ]
 [ 0.99123639]
 [ 0.99285167]
 [ 0.00907462]]
~~~

> save the trained model

~~~python
>>> import h5py
>>> model_structure = model.to_json()
 
>>> with open("basic_model.json", "w") as json_file:
...     json_file.write(model_structure)
 
>>> model.save_weights("basic_weights.h5")
~~~

### 5.1.8. Onward and deepward

> more about neural networks 
> 
> * Different activation functions (such as sigmoid, rectified linear units, and hyperbolic tangent)
> * Choosing a good learning rate, to dial up or down the effect of the error
> * Dynamically adjusting the learning rate using a momentum model to find the global minimum faster
> * Application of dropout, where a randomly chosen set of weights are ignored in a given training pass to prevent the model from becoming too attuned to its training set (overfitting)
> * Regularization of the weights to artificially dampen a single weight from growing or shrinking too far from the rest of the weights (another tactic to avoid overfitting)

### 5.1.9. Normalization: input with style

<b>`input normalization` as a common practice</b>

> * make each element retains its useful information from sample to sample
> * ensures that each neuron works within a similar range of input values as the other elements within a single sample vector

> approaches: 
> 
> * mean normalization
> * feature scaling
> * coefficient of variation

> the goal is to get the data in some range like [-1, 1] or [0, 1] for each element in each sample without losing information </br>
> TF-IDF, one-hot encoding, and word2vec (as you’ll soon see) are normalized already</br>
> 


## Appendix: Code