# Ch 9. Improving retention with long short-term memory networks

background for `lstm`: with `cnn` and `rnn`

> * for setence "The young `woman` `went` to the movies with her friends"
> 	
> 	`woman` and `went` can be captured by `cnn` and `rnn` 
> 
> * but for setence "The young `woman`, having found a free ticket on the ground
> 	
> 	`woman` and `went` can not be captured by `cnn` and `rnn` (the weights in `rnn` decay too quickly in time as you roll through each sentence)
> 
> these are the reason why `lstm` is used

reference: 

> * “[Long Short-Term Memory](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)” by Hochreiter and Schmidhuber in 1997
> * “[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation]( https://arxiv.org/pdf/1406.1078.pdf)”, by Kyunghyun Cho et al, 2014
> * Christopher Olah’s blog post explains why this is: [https://colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs)

## 9.1. LSTM

<b>`LSTM`</b>

> * can be trained to learn what to remember (the rules that govern the information stored in the `state (memory)` are trained neural nets themselves)
> * can begin to learn dependencies that stretch not just one or two tokens away, but across the entirety of each data sample
> 
> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig01_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig01_alt.jpg)

unroll a `LSTM`

> * in addition to the activation output to the next time-step like `rnn`, `a memory state` passes through time steps also has been added

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig02_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig02_alt.jpg)

a closer look at the `LSTM` cell

> * Instead of `weights plus activation function` in `rnn`, `LSTM Cell` introduces a `forget gate`, an `input/candidate gate` and an `output gate`
>
> * Figure 9.3. LSTM layer at time step t</br>
> 
>   ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig03_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig03_alt.jpg)

a simple example (by modifying code from previouse `rnn` example)

~~~python
>>> maxlen = 400
>>> batch_size = 32
>>> embedding_dims = 300
>>> epochs = 2
>>> from keras.models import Sequential
>>> from keras.layers import Dense, Dropout, Flatten, LSTM
>>> num_neurons = 50
>>> model = Sequential()
>>> model.add(LSTM(num_neurons, return_sequences=True,
...                input_shape=(maxlen, embedding_dims))) # change `SimpleRNN` to `LSTM`
>>> model.add(Dropout(.2))
>>> model.add(Flatten())
>>> model.add(Dense(1, activation='sigmoid'))
>>> model.compile('rmsprop', 'binary_crossentropy',  metrics=['accuracy'])
>>> print(model.summary())
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 400, 50)           70200
_________________________________________________________________
dropout_1 (Dropout)          (None, 400, 50)           0
_________________________________________________________________
flatten_1 (Flatten)          (None, 20000)             0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 20001
=================================================================
Total params: 90,201.0
Trainable params: 90,201.0
Non-trainable params: 0.0
~~~

> `LSTM` has many more parameters to train than `SimpleRNN` in last chapter
>
> * `SimpleRNN`: 351 * 50 = 17,550 weights for the layer
> 	* 300 (one for each element of the input vector)
> 	* 1 (one for the bias term)
> 	* 50 (one for each neuron’s output from the previous time step)
> * `LSTM`: three gates (a total of four neurons)
> 	* 17,550 * 4 = 70,200

<b>`LSTM` layer inputs</b>

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig04_alt.jpg)

> * send i<sup>th</sup> token of j<sup>th</sup> sample (300-dim wordvec) into 1<sup>st</sup> `LSTM` cell
> * `LSTM` concertrate: (1) `LSTM` output of (i-1)<sup>th</sup> token (50-dim vector, plus bias) (2) i<sup>th</sup> token (300-dim vector)

<b>`Forget Gate`</b>

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig05_alt.jpg)

> `forget gate` learns how much of the cell’s memory is going to be erased

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig06_alt.jpg)

> `forget gate` is just a feed forward network, with `n` neurons each with `m + n + 1` weights and `sigmoid activation function` to covert the output between 0 and 1

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig07_alt.jpg) 

> `forget gate` output a vector, used to update `memory vector` and erases elements of the memory vector (the closer it is to 0 the more of that memory value is erased)

<b>`candidate gate`</b>

> to learn how much to augment the memory, based on (1) `the input so far`; (2) `output from the last time step`

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/09fig08_alt.jpg)





### 9.1.1. Backpropagation through time

### 9.1.2. Where does the rubber hit the road?

### 9.1.3. Dirty data

### 9.1.4. Back to the dirty data

### 9.1.5. Words are hard. Letters are easier.

### 9.1.6. My turn to chat

### 9.1.7. My turn to speak more clearly

### 9.1.8. Learned how to say, but not yet what

### 9.1.9. Other kinds of memory

### 9.1.10. Going deeper