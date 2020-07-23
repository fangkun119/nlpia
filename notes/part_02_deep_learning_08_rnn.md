# Ch 8. Loopy (recurrent) neural networks (RNNs)

> `CNN` for `NLP`
> 
> * words that occurred in clusters could be detected together
> * if those words jostled a little bit in position, the network could be resilient to it
> * `cnn` capture ordering relationship by capturing localized relationships, but there’s another way

> Input layer -> Filters (卷积层）->  Hidden Layer (隐藏层) -> Prediction
> 
> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig01_alt.jpg)
> 
> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig02_alt.jpg)

> this network can 
> 
> * react to the cooccurrences of tokens 
> *  model the relationships between a data sample, as a whole, to its associated label
> 
> but the disvantages are
> 
> * it can not react to all the co-occurrences equally, regardless of whether they’re separated from each other by a long document or right next to each other 
> * don’t work with variable length documents very well
> * can not capture the unlikely semantic relationship between the words (such as strong negation and modifier (adjectives and adverb) tokens like “not” or “good.”)
> 

## 8.1. Remembering with recurrent networks

problem to solve: 

> * The <b>stolen</b> car sped into the arena.
> * The <b>clown(小丑)</b> car sped into the arena.
> 
> two different emotions may arise in the reader of these two sentences as the reader reaches the end of the sentence

`RNN~

> `Recurrent Neural Nets (RNNs)` enable neural networks to remember the past words within a sentence and can capture the different word (`stolen` and `clown`) when reach the end of the sentence.

structure of `RNN`

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig03.jpg)

`RNN` unrolled

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig04_alt.jpg)

Detailed explaination

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig05_alt.jpg)
> 
> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig06_alt.jpg)

Demonstraction with example

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig07_alt.jpg)

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig08_alt.jpg)


### 8.1.1. Backpropagation through time

problem to solve: only the last word will have a label for backpropagation

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig09_alt.jpg)

approach

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig10_alt.jpg) 

steps

> * Break each data sample into tokens
> * Pass each token into a feedforward net
> * Pass the output of each time step to the input of the same layer alongside the input from the next time step
> * Collect the output of the last time step and compare it to the label
> * Backpropagate the error through the whole graph, all the way back to the first input at time step 0

### 8.1.2. When do we update what? (detail about backpropagation of RNN)

#### 8.1.2.1 general situation in which you only care about the last time step

![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig10_alt.jpg) 

problem to solve: 

> the weights you’re updating aren’t a different branch of a neural network (each leg is the same network at different time steps)

solution: 

> 1. weight corrections are calculated at each time step but not immediately updated
> 
> 	* In a feedforward network, all `weight updates` would be `calculated` once all the gradients have been calculated for that input
> 	* But must `hold the updates` until you `go all the way back` in time, to time step 0 for that particular input sample 
> 
> 2. the gradient calculations are based on the values that the weights had when they contributed that much to the error 
>  
>	* the tricky part is 
>		* a weight at time step t contributed something to the error 1
>		* the same weight received a different input at time step t+t and therefore contributed a different amount to the error
>	* the steps are
> 		* figure out the various changes to the weights (as if they were in a bubble) at each time step
> 		* sum up the changes and apply the aggregated changes to each of the weights at the last phrase

#### 8.1.2.2 some situations in which you also care about the early time steps

example

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig11_alt.jpg) 

> for each time steps, there is a predict_vale and a label (such as brand-word recognize)

steps

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/08fig12_alt.jpg)
> 
> * backpropagation: as in the first example, the weight corrections are additive
> 	* backpropagate from the last time step all the way to the first, summing up what you’ll change for each weight
> 	* do the same with the error calculated at the second-to-last time step and sum up all the changes all the way back to t=0
> 	* ... (repeat)
> 	* until get all the way back down to time step 0
> 
> * update weights 
> 	*  <b>only after</b> you have calculated the proposed change in the weights for the <b>entire backpropagation step for all the time steps</b>
> 	* updating the weights earlier would “pollute” the gradient calculations 

### 8.1.3. Recap

> [https://livebook.manning.com/book/natural-language-processing-in-action/chapter-8/81](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-8/81)

### 8.1.4. There’s always a catch

problems: 

> * alrough weight number is small, but expensive to train, especially for sequences of any significant length, say 10 tokens
> * when the network goes deeper, the `vanishing gradient problem` and `exploding gradient problem` begin to happen. it is much seriouse for `RNN`

in <b>next chapter</b>, will solve these problems

### 8.1.5. Recurrent neural net with Keras

below is a rudimental version of the code script for `rnn`, and the problems of 8.1.4 will leave to next chapter

~~~python
>>> # 00 import libs
>>> import glob
>>> import os
>>> from random import shuffle
>>> from nltk.tokenize import TreebankWordTokenizer
>>> from nlpia.loaders import get_data
>>> word_vectors = get_data('wv')
>>> 
>>> # 01 functions
>>> # [1] data preprocess
>>> def pre_process_data(filepath):
...     """
...     Load pos and neg examples from separate dirs then shuffle them
...     together.
...     """
...     positive_path = os.path.join(filepath, 'pos')
...     negative_path = os.path.join(filepath, 'neg')
...     pos_label = 1
...     neg_label = 0
...     dataset = []
...     for filename in glob.glob(os.path.join(positive_path, '*.txt')):
...         with open(filename, 'r') as f:
...             dataset.append((pos_label, f.read()))
...     for filename in glob.glob(os.path.join(negative_path, '*.txt')):
...         with open(filename, 'r') as f:
...             dataset.append((neg_label, f.read()))
...     shuffle(dataset)
...     return dataset
>>> 
>>> # [2] tokenize and vectorize
>>> def tokenize_and_vectorize(dataset):
...     tokenizer = TreebankWordTokenizer()
...     vectorized_data = []
...     for sample in dataset:
...         tokens = tokenizer.tokenize(sample[1])
...         sample_vecs = []
...         for token in tokens:
...             try:
...                 sample_vecs.append(word_vectors[token])
...             except KeyError:
...                 pass
...         vectorized_data.append(sample_vecs)
...     return vectorized_data
>>> 
>>> # [3] extract label
>>> def collect_expected(dataset):
...     """ Peel off the target values from the dataset """
...     expected = []
...     for sample in dataset:
...         expected.append(sample[0])
...     return expected
>>> 
>>> # 02 generate X,y for train & test
>>> dataset = pre_process_data('./aclimdb/train')
>>> vectorized_data = tokenize_and_vectorize(dataset)
>>> expected = collect_expected(dataset)
>>> split_point = int(len(vectorized_data) * .8)
>>> x_train = vectorized_data[:split_point]
>>> y_train = expected[:split_point]
>>> x_test = vectorized_data[split_point:]
>>> y_test = expected[split_point:]
>>>
>>> # 03 hyperparameters 
>>> maxlen = 400    		# 400 tokens per example
>>> batch_size = 32 		# batches of 32
>>> embedding_dims = 300 	# word vectors are 300 elements long
>>> epochs = 2 				# run for 2 epochs
>>> 
>>> # 04 pad and truncate (as in example of perviouse chapter cnn)
>>> # rnn can handle variable length and do not need pad and truncate
>>> #   when input is variable length, output of recurrent layer will aslo be varied
>>> #     a 4-token input will output a sequence 4 elements long
>>> #     a 100-token sequence will produce a sequence of 100 elements
>>> # but in this case, the next layer required the input are of same length
>>> #   this is reason why pad_trunc is used below
>>> import numpy as np
>>> x_train = pad_trunc(x_train, maxlen)
>>> x_test  = pad_trunc(x_test, maxlen)
>>> x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
>>> y_train = np.array(y_train)
>>> x_test  = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
>>> y_test  = np.array(y_test)
>>> 
>>> # 05 initialize an empty Keras network
>>> from keras.models import Sequential
>>> from keras.layers import Dense, Dropout, Flatten, SimpleRNN
>>> num_neurons = 50
>>> model = Sequential()
>>> 
>>> # 06 add an rnn layer (here requeire parameter `maxlen` used in `pad and truncate step`
>>> # maxlen: 400     # sequence length
>>> # num_neurons: 50 # hidden layer neurons number, just an arbitarily choice
>>> # output: 
>>> #   if return_sequences = True , shape of return is 400 * 50, with val for each step 
>>> #   if return_sequences = False, shape of return is 50, with val for only last step
>>> model.add(SimpleRNN(
...    num_neurons, return_sequences=True,
...    input_shape=(maxlen, embedding_dims)))
>>>
>>> # 07 add the rest laysers
>>> model.add(Dropout(.2))    # prevent overfitting
>>> model.add(Flatten())      # dense layer require the vector is flattenned
>>> model.add(Dense(1, activation='sigmoid'))
~~~


## 8.2. Putting things together

~~~python
>>> # 08 compile model
>>> model.compile('rmsprop', 'binary_crossentropy',  metrics=['accuracy'])
Using TensorFlow backend.
>>> model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn_1 (SimpleRNN)     (None, 400, 50)           17550
_________________________________________________________________
dropout_1 (Dropout)          (None, 400, 50)           0
_________________________________________________________________
flatten_1 (Flatten)          (None, 20000)             0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 20001
=================================================================
Total params: 37,551.0
Trainable params: 37,551.0
Non-trainable params: 0.0
_________________________________________________________________
None
~~~

automate your hyperparameter selection

> * don’t stick to grid search for too long
> * random search is much more efficient (http://hyperopt.github.io/hyperopt/)
> * if you really want to be professional about it, you’ll want to try Bayesean optimization
>
> hyperparameter optimizer only gets one shot at it every few hours 

parameters number

> alghrough just a small network but have 37,551 parameters! That’s a lot of weights to update based on 20,000 training samples

unrolling netwrok: `17,550` parameters to train, but will be unrolling this net `400` time steps for each example

> * those 17,550 parameters keep the during all the 400 unrollings (remain as the value before) until all the backpropagations have been calculated 
> * the updates to the weights occur at once at the end of the sequence forward propagation and subsequent backpropagation

### 8.3. Let’s get to learning our past selves

~~~python
>>> # 09 fit model and save to file
>>> model.fit(x_train, y_train,
...           batch_size=batch_size,
...           epochs=epochs,
...           validation_data=(x_test, y_test))
Train on 20000 samples, validate on 5000 samples
Epoch 1/2
20000/20000 [==============================] - 215s - loss: 0.5723 -
acc: 0.7138 - val_loss: 0.5011 - val_acc: 0.7676
Epoch 2/2
20000/20000 [==============================] - 183s - loss: 0.4196 -
acc: 0.8144 - val_loss: 0.4763 - val_acc: 0.7820
 
>>> model_structure = model.to_json()
>>> with open("simplernn_model1.json", "w") as json_file:
...     json_file.write(model_structure)
>>> model.save_weights("simplernn_weights1.h5")
Model saved.
~~~

## 8.4. Hyperparameters

model hyperparameters

	maxlen         = 400 # trade off between `training time` and `noise`
	embedding_dims = 300 # dicated by Word2vec model used
	batch_size     = 32  # increase batch_size will reduces the number of times backpropagation 
	                     # trade off between `training time` and `risk of local minimum`
	epochs         = 2   # easy to test and tune, by simply running the training process again
                         # save the model to file will be helpful
                         # another alternative is use `EarlyStopping`
	num_neurons    = 50  # is an important parameter

drop out:

> if you feel the model is overfitting the training data but you can’t find a way to make your model simpler, you can always try increasing the Dropout(percentage)

example: try `num_neurons = 100` instead of `50` and check validation accuracy

~~~python
>>> num_neurons = 100
>>> model = Sequential()
>>> model.add(SimpleRNN(
...     num_neurons, return_sequences=True, input_shape=(maxlen,\
...     embedding_dims)))
>>> model.add(Dropout(.2))
>>> model.add(Flatten())
>>> model.add(Dense(1, activation='sigmoid'))
>>> model.compile('rmsprop', 'binary_crossentropy',  metrics=['accuracy'])
Using TensorFlow backend.
>>> model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn_1 (SimpleRNN)     (None, 400, 100)          40100
_________________________________________________________________
dropout_1 (Dropout)          (None, 400, 100)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 40000)             0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 40001
=================================================================
Total params: 80,101.0
Trainable params: 80,101.0
Non-trainable params: 0.0
_________________________________________________________________
>>> model.fit(x_train, y_train,
...           batch_size=batch_size,
...           epochs=epochs,
...           validation_data=(x_test, y_test))
Train on 20000 samples, validate on 5000 samples
Epoch 1/2
20000/20000 [==============================] - 287s - loss: 0.9063 -
acc: 0.6529 - val_loss: 0.5445 - val_acc: 0.7486
Epoch 2/2
20000/20000 [==============================] - 240s - loss: 0.4760 -
acc: 0.7951 - val_loss: 0.5165 - val_acc: 0.7824
>>> model_structure = model.to_json()
>>> with open("simplernn_model2.json", "w") as json_file:
...     json_file.write(model_structure)
>>> model.save_weights("simplernn_weights2.h5")
Model saved.
20000/20000 [==============================] - 240s - loss: 0.5394 -
acc: 0.8084 - val_loss: 0.4490 - val_acc: 0.7970
~~~

## 8.5. Predicting

predicting

~~~python
>>> sample_1 = "I hate that the dismal weather had me down for so long, when
 will it break! Ugh, when does happiness return? The sun is blinding and 
 the puffy clouds are too thin. I can't wait for the weekend."
 
>>> from keras.models import model_from_json
>>> with open("simplernn_model1.json", "r") as json_file:
...     json_string = json_file.read()
>>> model = model_from_json(json_string)
>>> model.load_weights('simplernn_weights1.h5')
 
>>> vec_list = tokenize_and_vectorize([(1, sample_1)])
>>> test_vec_list = pad_trunc(vec_list, maxlen)
>>> test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen,\
...     embedding_dims))
 
>>> model.predict_classes(test_vec)
array([[0]], dtype=int32)
~~~

> in this example: compared with `cnn` in last chapter, `rnn` are expensive to train but do not provide better prediction 
> 
> if we care about terms already appeared in the same sequence, `rnn` is still not optimal solution due to the `vanishing gradients` problem

### 8.5.1. Statefulness

`stateful` parameter of `SimpleRNN` layer

> `False` by default, </br>
> if tune it to `True`, it will: 
> 
> * pass the last example's last output into itself at the next time step along with the first token input 
> * suitable for modeling a large document that has been split into paragraphs or sentences for processing
> 
> detail: [https://livebook.manning.com/book/natural-language-processing-in-action/chapter-8/170](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-8/170)

### 8.5.2. Two-way street

bidirectional recurrent neural 

> example: "They wanted to pet the <b>dog</b> whose <b>fur</b> was brown"
> 
> * humans read the sentence in one direction but are capable of flitting back to earlier parts
> * humans can deal with information that isn’t presented in the best possible order

`bidirectional recurrent neural` allow your model to flit back across the input as well.

~~~python
>>> from keras.models import Sequential
>>> from keras.layers import SimpleRNN
>>> from keras.layers.wrappers import Bidirectional
 
>>> num_neurons = 10
>>> maxlen = 100
>>> embedding_dims = 300
 
>>> model = Sequential()
>>> model.add(Bidirectional(SimpleRNN(
...    num_neurons, return_sequences=True),\
...    input_shape=(maxlen, embedding_dims)))
~~~

detail: [https://livebook.manning.com/book/natural-language-processing-in-action/chapter-8/176](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-8/176)

### 8.5.3. What is this thing?

### Summary

> * In natural language sequences (words or characters), what came before is important to your model’s understanding of the sequence.
> * Splitting a natural language statement along the dimension of time (tokens) can help your machine deepen its understanding of natural language.
You can backpropagate errors in time (tokens), as well as in the layers of a deep learning network.
> * Because RNNs are particularly deep neural nets, RNN gradients are particularly temperamental, and they may disappear or explode.
> * Efficiently modeling natural language character sequences wasn’t possible until recurrent neural nets were applied to the task.
> * Weights in an RNN are adjusted in aggregate across time for a given sample.
> * You can use different methods to examine the output of recurrent neural nets.
> * You can model the natural language sequence in a document by passing the sequence of tokens through an RNN backward and forward simultaneously.

## Appendix：Code
