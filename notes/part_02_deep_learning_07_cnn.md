# Ch 7. Getting words in order with convolutional neural networks (CNNs)

Language's power not only in the words itself, but also exists in 

> * order and combination of words
> * meaning hidden beneath the words
> * intent and emotion
> * ......

## 7.1. Learning meaning

word relationships to each others
 
> * word order
> 
> 	~~~text
> 	The dog chased the cat.
>	The cat chased the dog.
> 	~~~
> 
> * word proximity 
> 
> 	~~~bash
>  # “shone” refers to the word “hull” 
> 	The ship's hull, despite years at sea, millions of tons of cargo, and two mid-sea collisions, shone like new.
> 	~~~

relationship patterns: 

> * spatially(空间序): relationships in the position of words, processed by fixed windows when modeling
> * temporarily(时序): the words and letters become time series data, extended for an unknown amount of time

`multilayer perceptron` can only discover patterns by relating weights to pieces of the input</br>
`cnn` and `rnn` is used to discover patterns about relations of the tokens spatially or temporally

## 7.2. Toolkit

Frameworks: heavily abstracted toolsets for building models from scratch

> * Theano: [http://deeplearning.net/software/theano/](http://deeplearning.net/software/theano/)
> * TensorFlow: [http://www.tensorflow.org](http://www.tensorflow.org)
> * PyTorch: [http://pytorch.org/](http://pytorch.org/)

Libraries to ease the use of these underlying architectures

> * `Lasagne` for `Theano`
> * `Skflow` for `TensorFlow`
> * `Keras` ([https://keras.io/](https://keras.io/)) for balance the friendly API and versatility (can use either TensorFlow or Theano as its backend)
> * also need the `h5py` package for saving the internal state of your trained mode

Some API often used in `Keras`

> * `Sequential()`: a class for a neural net abstraction, methods includes: 
> 	* `compile()`
> 	* `fit()`
> * `hyperparameters`: `epochs`, `batch_size`, and `optimizer`

## 7.3. Convolutional neural nets

> box sliding over a field (image, text, ..)

### 7.3.1. Building blocks

> by example of image processing with sliding windows
> 
> 1 parameter is window size, we commonly see a window size of three-by-three (3, 3) pixels

### 7.3.2. Step size (stride)

> the distance traveled during the sliding phase is a parameter, usually set to 1

### 7.3.3. Filter composition

Filters are composed of two parts:

> * A set of weights (exactly like the weights feeding into the neurons from chapter 5)
> * An activation function

As each filter slides over the image, one stride at a time (it pauses and takes a snapshot of the pixels it’s currently covering) 

> * The values of those pixels are then multiplied by the weight associated with that position in the filter
> * Most often this activation function is ReLU (rectified linear units)
> * Filter (a set of weights) 就是卷积核，以`ReLU`激活函数为例、可以表示为: `z`<sub>0</sub> = `max(x`<sub>i</sub> `* w`<sub>j</sub>`), 0)`

>	![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig05_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig05_alt.jpg)

> When the window is sliding: 
> 
>	![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig06_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig06_alt.jpg)

### 7.3.4. Padding

> by example of `Keras`
> 
> * `padding='valid'`: ignore that the output is slightly smaller 
> 	* downfall: the edge is undersampled when passing into each filter multiple times
> 		* for a large image, it is usually not a problem
> 		* for a tweet, loosing the first several words of each message can be a large impact
> * `padding='same'`: adding enough data to the input’s outer edges
> 	* downfall: adding potentially unrelated data to the input, which in itself can skew the outcome
> 
> but you won’t have use for that strategy in NLP applications, for it’s fraught with its own peril.

~~~python
>>> from keras.models import Sequential
>>> from keras.layers import Conv1D
 
>>> model = Sequential()
>>> model.add(Conv1D(filters=16,
                     kernel_size=3,
                     padding='same',
                      activation='relu',
                     strides=1,
                     input_shape=(100, 300)))
~~~

### 7.3.5. Learning

> about the concept of CNN training

## 7.4. Narrow windows indeed

> `CNN` for `NLP`:
> 
> > * `CNN` can be applied to `NLP` by using `word vectors` (also known as `word embeddings`)
> > * convolve one-dimensional filters over a one-dimensional input, such as a sentence
> > 	* relevant information is in the relative “horizontal” positions though
> > 	* unlike image cnn, "vertical" relevant are arbitary and should not been used
>
> sliding window:
> 
> > ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig07.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig07.jpg)
> 
> 1-D CNN with embedding: 
> 
> > ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig08_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig08_alt.jpg)
> > 
> > a sliding window (`D*3`weight matrix) scanning throw the `word vectors` 

### 7.4.1. Implementation in Keras: prepping the data

> dataset: 
> 
> * [https://ai.stanford.edu/%7eamaas/data/sentiment/](https://ai.stanford.edu/%7eamaas/data/sentiment/)
> * compiled for the 2011 paper Learning Word Vectors for Sentiment Analysis by tanford AI department
> * Maas, Andrew L. et al., Learning Word Vectors for Sentiment Analysis, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, June 2011, Association for Computational Linguistics

> pretrained word2vec model from google news: 
> 
> * [“GoogleNews-vectors-negative300.bin.gz - Google Drive”](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
> 


> code: 

~~~python
>>> # Step 0. import your Keras convolution tools
>>> import numpy as np
>>> from keras.preprocessing import sequence
>>> from keras.models import Sequential
>>> from keras.layers import Dense, Dropout, Activation
>>> from keras.layers import Conv1D, GlobalMaxPooling1D
>>> 
>>> # Step 1. 加载样本
>>> # load data set (one file for one example) and then suffle
>>> # positive examples are in the `pos` folder
>>> # negative examples are in the `neg` folder
>>> import glob
>>> import os
>>> 
>>> from random import shuffle
>>> def pre_process_data(filepath):
...     """
...     This is dependent on your training data source but we will
...     try to generalize it as best as possible.
...     """
...
...     positive_path = os.path.join(filepath, 'pos')
...     negative_path = os.path.join(filepath, 'neg')
...     pos_label = 1
...     neg_label = 0
...     dataset = []
...
...     for filename in glob.glob(os.path.join(positive_path, '*.txt')):
...         with open(filename, 'r') as f:
...             dataset.append((pos_label, f.read()))
...     for filename in glob.glob(os.path.join(negative_path, '*.txt')):
...         with open(filename, 'r') as f:
...             dataset.append((neg_label, f.read()))
...
...     shuffle(dataset)
...     return dataset
>>> 
>>> dataset = pre_process_data('<path to your downloaded file>/aclimdb/train')
>>> dataset[0]
(1, 'I, as a teenager really enjoyed this movie! Mary Kate and Ashley worked
 great together and everyone seemed so at ease. I thought the movie plot was
 very good and hope everyone else enjoys it to! Be sure and rent it!! Also 
they had some great soccer scenes for all those soccer players! :)')
>>> 
>>> # Step 2. 提取样本特征
>>>  # tokenize and vectornize the data
>>> from nltk.tokenize import TreebankWordTokenizer
>>> from gensim.models.keyedvectors import KeyedVectors
>>> from nlpia.loaders import get_data
>>> word_vectors = get_data('w2v', limit=200000)
>>> 
>>> def tokenize_and_vectorize(dataset):
...     tokenizer = TreebankWordTokenizer()  # tokenizer
...     vectorized_data = []
...     expected = []
...     for sample in dataset:
...         tokens = tokenizer.tokenize(sample[1]) # tokenize
...         sample_vecs = []
...         for token in tokens:
...             try:
...						word_vec = word_vectors[token] # word_vec of this token
...                 sample_vecs.append(word_vectors[token]) # list<word_vec>
...             except KeyError:
...						# Google News Word2vec vocabulary includes some stopwords, but not all 
...						# this is not ideal by any stretch, but this will give a baseline for cnn
...                 pass  # no matching token in the Google w2v vocab
...         vectorized_data.append(sample_vecs) list<list<word_vec>>
...     return vectorized_data
>>> vectorized_data = tokenize_and_vectorize(dataset)
>>> 
>>> # Step 3. 提取样本标签
>>> # collect the target values—0 for a negative review, 1 for a positive review
>>> def collect_expected(dataset):
...     """ Peel off the target values from the dataset """
...     expected = []
...     for sample in dataset:
...         expected.append(sample[0])
...     return expected
>>> expected = collect_expected(dataset)
>>> 
>>> # Step 4. train test split
>>> split_point = int(len(vectorized_data)*.8)
 
>>> x_train = vectorized_data[:split_point_]
>>> y_train_ = expected[:split_point]
>>> x_test = vectorized_data[split_point:]
>>> y_test = expected[split_point:]
>>> 
>>> # Step 5. CNN参数
>>> maxlen = 400
>>> batch_size = 32
>>> embedding_dims = 300
>>> filters = 250
>>> kernel_size = 3
>>> hidden_dims = 250
>>> epochs = 2
>>> 
>>> # Step 6. 处理样本字数不同的问题
>>> # Keras' padding function `pad_sequences` only works with list<scalar>, not list<word_vector>, we write the padding funciton by our own
>>> def pad_trunc(data, maxlen):
...     """
...     For a given dataset pad with zero vectors or truncate to maxlen
...     """
...     # 存放计算结果
...     new_data = []
...
...     # 创建用于padding的zero word vector
...     # create a vector of 0s the length of our word vectors
...     zero_vector = []
...     for _ in range(len(data[0][0])):
...         zero_vector.append(0.0)
...
...     # 遍历每一个样本
...     for sample in data:
...         if len(sample) > maxlen:
...             # 超长的直接做截断
...             temp = sample[:maxlen]
...         elif len(sample) < maxlen:
...             # 长度不够的，用零向量（zero word vector）填充
...             # Append the appropriate number 0 vectors to the list
...             temp = sample
...             additional_elems = maxlen - len(sample)
...             for _ in range(additional_elems):
...                 temp.append(zero_vector)
...         else:
...	             # 长度刚好的不做处理
...             temp = sample
...         new_data.append(temp)
...     return new_data
>>> x_train = pad_trunc(x_train, maxlen)
>>> x_test  = pad_trunc(x_test, maxlen)
>>> 
>>> # Step 7: padding-truncate整形之后的数据集，<样本数，统一之后的样本token数，token word vector维度数>
>>> x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
>>> x_test  = np.reshape(x_test, (len(x_test),   maxlen, embedding_dims))
>>> y_train = np.array(y_train)
>>> y_test  = np.array(y_test)
~~~

### 7.4.2. Convolutional neural network architecture

~~~python
>>> # Step 8: 创建一个卷积层
>>> # Construct a 1D CNN
>>> print('Build model...')
>>> model = Sequential()
 
>>> model.add(Conv1D(
...    filters,						# 卷积核(filter)数量：250 
...    kernel_size,					# silding window size为3
...    padding='valid',				# silding window 使用vadid padding
...    activation='relu',			# 激活函数使用ReLU
...    strides=1,					# sliding window 步长为1
...    input_shape=(maxlen, embedding_dims) # 样本特征格式：归一化之后的字数 * word vector维度数
... ))
~~~

### 7.4.3. Pooling

<b>effect of pooling:</b>

1. pooling is for dimensionality reduction and learning higher order representations of the source data
 
	> * the filters (卷积核）are being trained to find patterns. The patterns are revealed in relationships between words and their neighbors
	> 	* in image processing, 
	> 		* the first layers will tend to learn to be edge detectors, places where pixel densities rapidly shift from one side to the other
	> 		* later layers learn concepts like shape and texture. 
	> 		* layers after that may learn “content” or “meaning.” Similar processes will happen with text

2. computational savings
3. location invariance 

	> if an original input element is jostled slightly in position in a similar but distinct input sample, the max pooling layer will still output something similar
	> 

<b>choices of pooling</b>

> * `average pooling`: by taking the average of the subset of values
> * `max pooling`: by taking the largest activation value for the given region

> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/07fig09_alt.jpg)

~~~python
>>> # Step 9: 池化层
>>> model.add(GlobalMaxPooling1D())
~~~

<b>`GlobalMaxPooling1D layer`</b>: 

> * it's just a very crude（粗糙的） model
> * instead of taking the max of a `small subsection` of each filter’s output, you’re taking the `max of the entire output` for that filter, which results in a large amount of information loss
> * but even tossing lose all that good information, this toy model won’t be deterred


### 7.4.4. Dropout

`dropout` is used to prevent overfitting

> the idea is that on each training pass, if you “turn off” a certain percentage of the input going to the next layer, randomly chosen on each pass, the model will be less likely to learn the specifics of the training set, “overfitting”

~~~python
>>> # Step 10. 增加使用了drop out的隐藏层
>>> model.add(Dense(hidden_dims))
>>> model.add(Dropout(0.2))
>>> model.add(Activation('relu'))
~~~

> a 20% dropout setting is common, but a dropout of up to 50% can have good results (one more hyperparameter you can play with)

### 7.4.5. The cherry on the sundae（add output layer, compile and train the cnn model)

~~~python
>>> # Step 11. outpuut layer
>>> model.add(Dense(1))
>>> model.add(Activation('sigmoid'))
>>> 
>>> # Step 11. compile the cnn 
>>> model.compile(loss='binary_crossentropy', # for multiple type classification, could use 'categorical_crossentropy'
...               optimizer='adam',  #  Adam, RSMProp, ...
...               metrics=['accuracy'])
>>>
>>> # Step 12. train model
>>> # other parameters includes `checkpointing`, `EarlyStopping`, ...
>>> model.fit(x_train, y_train,
...           batch_size=batch_size,
...           epochs=epochs,
...           validation_data=(x_test, y_test))   # it's validate set 
Using TensorFlow backend.
Loading data...
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
x_train shape: (25000, 400)
x_test shape: (25000, 400)
Build model...
Train on 20000 samples, validate on 5000 samples
Epoch 1/2 [================================] - 417s - loss: 0.3756 -
acc: 0.8248 - val_loss: 0.3531 - val_acc: 0.8390
Epoch 2/2 [================================] - 330s - loss: 0.2409 -
acc: 0.9018 - val_loss: 0.2767 - val_acc: 0.8840
~~~


### 7.4.6. Let’s get to learning (more trick on model training)

> save model file

~~~python
>>> # Step 13. save model
>>> model_structure = model.to_json()
>>> with open("cnn_model.json", "w") as json_file:
...     json_file.write(model_structure)
>>> model.save_weights("cnn_weights.h5")
~~~

> for debugging purpose, the `random seed` can be freeze like below

~~~python
>>> import numpy as np
>>> np.random.seed(1337)
~~~

> understanding the effect of validate set

~~~
validation_data=(x_test, y_test)
~~~

### 7.4.7. Using the model in a pipeline

~~~python
>>> # Step 14. load model
>>> from keras.models import model_from_json
>>> with open("cnn_model.json", "r") as json_file:
...     json_string = json_file.read()
>>> model = model_from_json(json_string) 
>>> model.load_weights('cnn_weights.h5')
>>> 
>>> # Step 15. test example
>>> sample_1 = "I hate that the dismal weather had me down for so long, 
 when will it break! Ugh, when does happiness return? The sun is blinding
 and the puffy clouds are too thin. I can't wait for the weekend."
>>> 
>>> # Step 16. genrate feature and predict
>>> vec_list = tokenize_and_vectorize([(1, sample_1)])
>>> test_vec_list = pad_trunc(vec_list, maxlen)
>>> test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen,\
...     embedding_dims))
>>> model.predict(test_vec) # score
array([[ 0.12459087]], dtype=float32)
>>> model.predict_classes(test_vec) # class
array([[0]], dtype=int32)

~~~

### 7.4.8. Where do you go from here?

> [https://livebook.manning.com/book/natural-language-processing-in-action/chapter-7/222](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-7/222)

### Summary

> * A convolution is a window sliding over something larger (keeping the focus on a subset of the greater whole).
> * Neural networks can treat text just as they treat images and “see” them.
> * Handicapping the learning process with dropout actually helps.
> * Sentiment exists not only in the words but in the patterns that are used.
> * Neural networks have many knobs you can turn.

## Appendix: Code