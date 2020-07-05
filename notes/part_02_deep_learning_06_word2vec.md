# Ch 06. Reasoning with word vectors (Word2vec)

> In the previous chapters, we ignore: 
> 
> * nearby context of a word
> * words around each word
> * effect the neighbors of a word have on its meaning and how those relationships affect the overall meaning of a statement

> `bow` (bag-of-words)
> 
> *  jumbled all the words from each document together into a statistical bag

> `word vectors` will be able to: 
> 
> * identify synonyms, antonyms, or words that just belong to the same category (such as people, anaminal, places, plants, names, or concepts) 
> * "latent semantic analysis" also can do this, but there is a tighter limit of word neighbour hood thus lead to a tighter accuracy

## 6.1. Semantic queries and analogies

example: "She invented something to do with physics in Europe in the early 20th century" : "Marie Curie"

> with `word vectors`, you can search for words or names that combine the meaning of the words “woman,” “Europe,” “physics,” “scientist,” and “famous”

~~~python
>>> # wv: short for word_vector
>>> answer_vector = wv['woman'] + wv['Europe'] + wv[physics'] + wv['scientist']
~~~

> we can even subtract gender bias (take the “man” out of “woman”)

~~~python
>>> answer_vector = wv['woman'] + wv['Europe'] + wv[physics'] + wv['scientist'] - wv['male'] - 2 * wv['man']
~~~

### 6.1.1. Analogy questions

example: "Who is to nuclear physics what Louis Pasteur is to germs?" 

~~~python
>>> answer_vector = wv['Louis_Pasteur'] - wv['germs'] + wv['physics']
~~~

example: "Marie Curie is to science as who is to music?"

~~~python
>>> # MARIE CURIE : SCIENCE :: ? : MUSIC
>>> wv['Marie_Curie'] - wv['science'] + wv['music']
~~~

for `Google’s pretrained word vector model`

> your word is almost certainly within the 100B word news feed that Google trained it on, unless your word was invented after 2013.

## 6.2. Word vectors

word vectors: 

> * a neural network trained by predicting word occurrences near each target word:
> 	* the prediction is merely a means to an end. 
> 	* what you do care about is the internal representation (the vector that Word2vec gradually builds up to help it generate those predictions)
> 	* this representation will capture much more of the target word (its semantics) than the `word-topic` vectors in chapter 04
> * typically have 100 to 500 dimensions (depending on the breadth of information in the corpus used to train them)

`Word2vec` learns about things you might not think to associate with all words, such as: 

> * `geography`, `sentiment (positivity)`, and `gender` associated with a word
> * `placeness`, `peopleness`, `conceptness` or `femaleness` degree of a word 
> * the meaning of a word “rubs off” on the neighboring words

`Word2vec` vector compared with `topic vector` of chapter 04

> * numerical vectors
> * word vector means something more specific, more precise
> 	* for `LSA`, words only had to occur in the same document
> 	* for `Word2vec` word vectors, the words must occur near each other—typically fewer than five words apart and within the same sentence
> * word vector topic weights can be added and subtracted to create new word vectors that mean something

to understanding `Word2Vec` (by a mental model): 

> just think of word vectors as a list of weights or scores

~~~python
>>> from nlpia.book.examples.ch06_nessvectors import *
>>> # nessvector means vector of all kinds of *ness
>>> # nessvecotr('Marie_Curie') is a vector of a word
>>> nessvector('Marie_Curie').round(2)
placeness     -0.46
peopleness     0.35
animalness     0.17
conceptness   -0.32
femaleness     0.26
~~~

> you can compute “nessvectors” for any word or `n-gram` in the `Word2vec vocabulary`</br>
> tool: [https://github.com/totalgood/nlpia/blob/master/src/nlpia/book/examples/ch06_nessvectors.py](https://github.com/totalgood/nlpia/blob/master/src/nlpia/book/examples/ch06_nessvectors.py)

vector-oriented reasoning: 

> do math with word vectors and that the answer makes sense when you translate the vectors back into words

~~~python
# for those not up on sports, the Portland Timbers and Seattle Sounders are major league soccer teams
wv['Timbers'] - wv['Portland'] + wv['Seattle'] = ?

# ideally you’d like this math (word vector reasoning) to give you this:
wv['Seattle_Sounders'] 
~~~

similarly

~~~python
#  "'Marie Curie’ is to ‘physics’ as __ is to ‘classical music’?"
wv['Marie_Curie'] - wv['physics'] + wv['classical_music'] = ?
~~~

### 6.2.1. Vector-oriented reasoning

references: </br>

> “[Linguistic Regularities in Continuous Space Word Representations](https://www.aclweb.org/anthology/N13-1090),” by Tomas Mikolov, Wentau Yih, and Geoffrey Zweig </br>
> "[Radim Řehů řek’s interview of Tomas Mikolov](https://rare-technologies.com/rrp#episode_1_tomas_mikolov_on_ai)"</br>
> "[ICRL2013 open review](https://openreview.net/forum?id=idpCdOWtqXd60&noteId=C8Vn84fqSG8qa)"

word-vector reasoning example: </br>

> question like 
> 
> ~~~python
> Portland Timbers + Seattle - Portland = ?
> ~~~
> 
> can be solved with vector algebra as below
> 
> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig01_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig01_alt.jpg)
> 
> `Word2vec model` contains information about the `relationships between words`, including `similarity`， the model knowns: 
> 
> * `Portland` and `Portland Timbers` are roughly the <b>same distance</b> apart as `Seattle` and `Seattle Sounders` 
> * These two distances are roughly in the same direction
> 
> thus `Portland Timbers + Seattle - Portland`  is close to the term of is close to `Seattle Sounders`

other usages example: 

> * transform `token occurrence counts or frequencies vectors` into `Word2vec vectors` (with much lower-dimensions)
> * discovered that the difference between a `singular`(单数) and a `plural`(复数) word 
> 	![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0189-01_alt.jpg)
> * answer questions that involve geography, culture, and demographics, for example 
>
> 	~~~text
> 	# "San Francisco is to California as what is to Colorado?"
> 	San Francisco - California + Colorado = Denver
> 	~~~
> 
> * word vectors for ten US cities projected onto a 2D map 
> 
> 	![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig02_alt.jpg)
> 
> * overcome some of the rigidity, brittleness of pattern, or keyword matching
> 
> 	for example, search for a famouse person in `Houston` (but he/she has moved to `Dallas`) by reference of `word vector` model behind the map as above 

### 6.2.2. How to compute Word2vec representations

#### 6.2.2.1 Introduction

2 approaches: 

> * `skip-gram approach` predicts the `context of words (output words)` from `the target word (the input word)`
> * `continuous bag-of-words (CBOW) approach` predicts the `target word (the output word)` from `the nearby words (input words)`

this will be resource intensive, but pretrained representations can be relied: 

> pretraind models of `Word2Vec`, `GloVe`, `FastText` are presented as below 
> 
> * pretrained `word vector` model from corpus of Wikipedia, DBPedia, Twitter, and Freebase: 
>
> 	[GitHub - 3Top/word2vec-api: Simple web service providing a word embedding model](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-model)
> * pretrained Word2vec model based on English Google News articles: 
> 
> 	[Original Google 300-D Word2vec model on Google Drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM) 
> 
> * FastText word model for 294 languages by Facebook
> 
> 	[GitHub - facebookresearch/fastText: Library for fast text representation and classification](https://github.com/facebookresearch/fastText)

pretrained model not always suitable

> If you need to constrain your word vectors to their usage in a particular domain, you’ll need to train them on text from that domain

#### 6.2.2.2 Skip-gram approach

<b>approach</b>: predict the `surrounding window of words` based on an `input word` 

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig03_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig03_alt.jpg)
>
> <b>w<sup>t</sup></b>: the one-hot vector for the token at position `t` 
> 

<b>network structure</b>

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig04_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig04_alt.jpg)
> 
> the input word is “Monet,” and the expected output of the network is either “Claude” (the 1st network diagram) or “painted” (the 2nd network diagram) 

<b>output value</b>: 

> for each of the K output nodes, the softmax output value can be calculated using the normalized exponential function
> 
> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0192-01.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0192-01.jpg) 
> 
> for example: if your output vector of a three-neuron output layer looks like this
> 
> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0192-02.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0192-02.jpg)
> 
> the “squashed” vector after the softmax activation would look like: 
> 
> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0193-01.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0193-01.jpg) 
> 
> the sum of these values (rounded to three significant digits) is approximately 1.0, like a probability distribution

<b>How does `skip-gram` approach learn the vector?</b>

> 1. one input example: a sentence
> 
> 	~~~python
> 	>> sentence = "Claude Monet painted the Grand Canal of Venice in 1806."
> 	~~~
> 
> 2. parse the senetence to 10 `5-grams` with the input word at the center
> 
> 	![](https://dpzbhybb2pdcj.cloudfront.net/lane/HighResolutionFigures/table_6-1.png)
> 
> 3. for each input word (center word), there are 4 training iterations (for 4 different neural weights), where each iteration predict one output word (surrounding word)
> 
> 4. after the training
> 
> * semantically similar words will have similar vectors, because they were trained to predict similar surrounding words

<b>retrieving word vectors with linear algebra</b>

> `weights of a hidden layer` (word embedding): one column per input neuron, one row per output neuron
> 
> * the output layer of the network can be ignored. Only the weights of the inputs to the hidden layer are used as the embeddings
> * the dot product between the `one-hot vector representing the input term` and `the weights` then `represents the word vector embedding`
>
> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig05_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig05_alt.jpg)

#### 6.2.2.3 Continuous bag-of-words approach

<b>approach</b>: predict the `center word` based on the `surrounding words` 

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig06_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig06_alt.jpg)

> it won't create pairs of input and output tokens (as what has been done in `skip-gram` appraoch), it will create a multi-hot vector of all surrounding terms as an input vector


<b>input</b>: the multi-hot vector 

> the sum of the one-hot vectors of the surrounding words’ training pairs w<sub>t-2</sub> + w<sub>t-1</sub> + w<sub>t+1</sub> + w<sub>t+2</sub>

<b>the network</b>: 

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig07_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig07_alt.jpg)

<b>terminoloty</b>: continuous bag of words

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0197-01_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0197-01_alt.jpg)

#### 6.2.2.4 Skip-gram vs. CBOW: when to use which approach

`skip-gram approach`: 

> * works well with small corpora and rare terms
> * because it has more examples due to the network structure

`continuous bag-of-words`: 

> * higher accuracies for frequent words 
> * much faster to train

#### 6.2.2.5 Frequent bigrams 

##### (1) consider `frequent bigrams` to be a `single term`

the team used co-occurrence frequency to identify bigrams and trigrams that should be considered single terms

> for example, consider bigrams “New York” and “San Francisco” as single terms “New\_York” and “San\_Francisco” 

`bigram scroing function` to find the frequent bigrams
 
> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0198-01.jpg)

##### (2) Subsampling frequent tokens

examples: 

> * common words like “the” or “a” often don’t carry significant information
> * the co-occurrence of the word “the” with a broad variety of other nouns in the corpus might create less meaningful connections between words

approach: 

> to reduce the emphasis on frequent words like stop words, words are sampled during training in inverse proportion to their frequence

sampling probability: 

> to determines whether or not a particular word is included in a particular skip-gram during training

> * in `Tomas Mikolov`'s paper
> 
> 	![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0199-01.jpg)
>
> * in Word2vec C++ implementation (different sample probability but has the same effect)
> 
> 	![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0199-02.jpg) 
> 
> 	* `f(w`<sub>`i`</sub>`)`: represents the frequency of a word across the corpus
> 	* `t`: represents a frequency threshold above which you want to apply the subsampling probability, 
> 		* it depends on: 
> 			* corpus size, 
> 			* average document length, 
> 			* the variety of words used in those documents. 
> 		* values between 10<sup>-5</sup> and 10<sup>-6</sup> are often found in the literature
>		* example: if a word shows up 10 times across your entire corpus, and your corpus has a vocabulary of one million distinct words, and you set the subsampling threshold to 10-6, the probability of keeping the word in any particular n-gram is 68%. You would skip it 32% of the time while composing your n-grams during tokenization.

##### (3) Negative sampling

usage:

> `negative sampling` is used to speed up the training of word vector models, since `a single training example` will update all weights of the network and vocabulary size are large

approach:

> * instead of updating all word weights not in the window, this approach sampls just a few negative samples (in the output vector) to update their weights
> * instead of updating all weights, it just pick n negative example word pairs (words that don’t match your target output for that example) and update the weights that contributed to their specific output.

> the computation can be reduced dramatically and the performance of the trained network doesn’t decrease significantly

### 6.2.3. How to use the `gensim.word2vec` module

<b>download the pretrained word2vec model</b>

> download a pretrained word2vec model

~~~python
>>> from nlpia.data.loaders import get_data
>>> word_vectors = get_data('word2vec')
~~~

> or download original model trained by Mikolov hosted on Google Drive at [https://bit.ly/GoogleNews-vectors-negative300](https://bit.ly/GoogleNews-vectors-negative300)

~~~python
>>> # load the whole model
>>> from gensim.models.keyedvectors import KeyedVectors
>>> word_vectors = KeyedVectors.load_word2vec_format(\
...     '/path/to/GoogleNews-vectors-negative300.bin.gz', binary=True)

>>> # or only load the 200k most common words since it is very large
>>> from gensim.models.keyedvectors import KeyedVectors
>>> word_vectors = KeyedVectors.load_word2vec_format(\
...     '/path/to/GoogleNews-vectors-negative300.bin.gz',
...         binary=True, limit=200000)
~~~

<b>notice:</b> `word vector` model with a `limited vocabulary` will lead to a `lower performance `

> when testing documents contain new words that you haven’t been loaded into the word vectors before, the `word vector` will perform pool

> guidelines: 
> 
> * limit word size only when writing and debuging the code 
> * once the code has been completed, the application should use the `complete Word2vec model` 

<b>query the model:</b>

~~~python
>>> # cooking + potatoes
>>> word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5)
[('cook', 0.6973530650138855),
 ('oven_roasting', 0.6754530668258667),
 ('Slow_cooker', 0.6742032170295715),
 ('sweet_potatoes', 0.6600279808044434),
 ('stir_fry_vegetables', 0.6548759341239929)]
>>> word_vectors.most_similar(positive=['germany', 'france'], topn=1)
[('europe', 0.7222039699554443)]
>>> 
>>> # perform calculations (such as the famous example king + woman - man = queen)
>>> word_vectors.most_similar(positive=['king', 'woman'],
...     negative=['man'], topn=2)
[('queen', 0.7118192315101624), ('monarch', 0.6189674139022827)]
~~~

> * `positive`: a list of the word-vectors to be added together
> * `negative`: a list of the word-vectors to be subtracted and to exclude unrelated terms
> * `topn`: how many related terms should be provided as a return value
> `word2vec` synonomy (similarity) is a continuous score, a distance

~~~python
>>> word_vectors.doesnt_match("potatoes milk cake computer".split())
'computer'
~~~

> * `doesnt_match`: find un-related term (the term with the highest distance to all other list terms)

~~~python
>>> word_vectors.similarity('princess', 'queen')
0.70705315983704509
~~~

> * `word_vectors.similarity`: calculate the similarity between two terms

~~~python
>>> word_vectors['phone']
array([-0.01446533, -0.12792969, -0.11572266, -0.22167969, -0.07373047,
       -0.05981445, -0.10009766, -0.06884766,  0.14941406,  0.10107422,
       -0.03076172, -0.03271484, -0.03125   , -0.10791016,  0.12158203,
        0.16015625,  0.19335938,  0.0065918 , -0.15429688,  0.03710938,
        ...
~~~

> * `[]` or `get()` method: retrieve the raw word vector for self-defined calculation

### 6.2.4. How to generate your own word vector representations

<b>usage</b>

> for creating your own domain-specific word vector models
> 
> * doing this need a lot of documents to do this as well as Google and Mikolov did
> * but if 
> 	* your words are particularly rare on Google News
> 	* your texts use them in unique ways within a restricted domain
> 	* such as medical texts or transcripts, 
> 	* a domain-specific word model may improve your model accuracy

<b>preprocessing</b>

> 1. break your documents into sentences</br>
> 	  [Detector Morse](https://github.com/cslu-nlp/DetectorMorse): segmenter trained with WSJ journals
> 2. break down the sentences into tokens

~~~python
>>> token_list
[
  ['to', 'provide', 'early', 'intervention/early', 'childhood', 'special',
   'education', 'services', 'to', 'eligible', 'children', 'and', 'their',
   'families'],
  ['essential', 'job', 'functions'],
  ['participate', 'as', 'a', 'transdisciplinary', 'team', 'member', 'to',
   'complete', 'educational', 'assessments', 'for']
  ...
]
~~~

<b>train your domain-specific Word2vec model</b>

~~~python
>>> # lib
>>> from gensim.models.word2vec import Word2Vec
>>> 
>>> # hyper parameters
>>> num_features 	= 300	
>>> min_word_count 	= 3		
>>> num_workers 		= 2		
>>> window_size 		= 6		
>>> subsampling 		= 1e-3	
>>>
>>> # train model
>>> model = Word2Vec(
...     token_list,
...     workers		= num_workers,
...     size			= num_features,
...     min_count	= min_word_count,
...     window		= window_size,
...     sample		= subsampling
... )
>>>
>>> # reduce the memory usage
>>> # freeze the model, storing the weights of the hidden layer and discarding the output weights
>>> model.init_sims(replace=True)
>>> 
>>> # save the model to file
>>> model_name = "my_domain_specific_word2vec_model"
>>> model.save(model_name)
>>> 
>>> # loading a saved model for calculation on other words
>>> from gensim.models.word2vec import Word2Vec
>>> model_name = "my_domain_specific_word2vec_model"
>>> model = Word2Vec.load(model_name)
>>> model.most_similar('radiology')
~~~

### 6.2.5. Word2vec vs. GloVe (Global Vectors)

Approaches

> `word2vec` is based on `neural network` with backpropagation, which is less efficient than direct optimization of a cost function using gradient descent</br>
> `GloVe`: 
> 
> * counting the word co-occurrences and recording them in a square matrix
> * compute the singular value decomposition of this co-occurrence matrix
> * splitting it into the same two weight matrices that Word2vec produces
> 
> the key was to normalize the co-occurrence matrix the same way

> in some cases the Word2vec model failed to converge to the same global optimum that the Stanford researchers were able to achieve with their SVD approach

Advantages: 

> * GloVe can produce matrices equivalent to word2vec, with same accuracy but in much less time
> * GloVe can be trained on smaller corpora and still converge 
> * Faster training
> * Better RAM/CPU efficiency (can handle larger documents)
> * More efficient use of data (helps with smaller corpora)
> * More accurate for the same amount of training

Reference:

> [Stanford GloVe Project](https://nlp.stanford.edu/projects/glove/) </br>
> SVD: chapter 5 and Append C of this book </br>
> [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), by Jeffrey Pennington, Richard Socher, and Christopher D. Manning </br>
> [24 Gensim’s comparison of Word2vec and GloVe performance](https://rare-technologies.com/making-sense-of-Word2vec/#glove_vs_word2vec) </br>

### 6.2.6. fastText

Approaches: 

> `FastText` predicts the surrounding n-character grams rather than just the surrounding words, like `Word2vec` does

> for examples: 
>
> ~~~text
> # the word “whisper” would generate the following 2- and 3-character grams:
> wh, whi, hi, his, is, isp, sp, spe, pe, per, er
> ~~~


> `FastText` trains a vector representation for every n-character gram, which includes words, misspelled words, partial words, and even single characters 

Advantages: 

> * it handles rare words much better than the original Word2vec approach

Pretrained Model (in 294 languages): 

> [fastText/pretrained-vectors.md at master](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

Use the pretrained fastText models: 

> download the pretrained model file, unzip it, and load it with `gensim`

~~~python
>>> from gensim.models.fasttext import FastText
>>> ft_model = FastText.load_fasttext_format(\
...     model_file=MODEL_PATH)
>>> ft_model.most_similar('soccer')
~~~

### 6.2.7. Word2vec vs. LSA

LSA topic-document vectors

> * are the sum of the `topic-word vectors` for all the `words` in those documents
> * faster than `Word2vec`
> * for long documents, it does a better job of discriminating and clustering those documents

`Word2vec` vectors 

> * can be created using the exact same `SVD` algorithm used for `LSA`
> * but `Word2vec` gets more use out of the same number of words by creating a sliding window
> * can do semantic reasoning 

incremental or online training 

> * both these 2 approaches can only account for the co-occurrences terms in the new documents (only weights of existing bins in the lexicon can be updated)
> * adding completely new words would change the total size of your vocabulary and therefore your one-hot vectors would change

domain-specific `Word2vec` models

> * Harry Potter for example: [https://github.com/nchah/word2vec4everything#harry-potter](https://github.com/nchah/word2vec4everything#harry-potter)

Summary: 

> Advantages of LSA are
>
> * Faster training
> * Better discrimination between longer documents
>
> Advantages of Word2vec and GloVe are
> 
> * More efficient use of large corpora
> * More accurate reasoning with words, such as answering analogy questions

### 6.2.8. Visualizing word relationships

#### 6.2.8.1 query a word2vec embedding

~~~python
>>> # load the word2vec embedding
>>> import os
>>> from nlpia.loaders import get_data
>>> from gensim.models.word2vec import KeyedVectors
>>> wv = get_data('word2vec')
>>> len(wv.vocab)
3000000

>>> # get word with the index location of a word
>>> import pandas as pd
>>> vocab = pd.Series(wv.vocab)
>>> vocab.iloc[1000000:100006]
Illington_Fund            Vocab(count:447860, index:2552140)
Illingworth               Vocab(count:2905166, index:94834)
Illingworth_Halifax       Vocab(count:1984281, index:1015719)
Illini                    Vocab(count:2984391, index:15609)
IlliniBoard.com           Vocab(count:1481047, index:1518953)
Illini_Bluffs             Vocab(count:2636947, index:363053)

>>> # get the 300-D vector of a perticular word
>>> wv['Illini']
array([ 0.15625   ,  0.18652344,  0.33203125,  0.55859375,  0.03637695,
       -0.09375   , -0.05029297,  0.16796875, -0.0625    ,  0.09912109,
       -0.0291748 ,  0.39257812,  0.05395508,  0.35351562, -0.02270508,
       ...
~~~

> * compound words and common n-grams are joined together (with an underscore character ("\_"))

~~~python
>>> import numpy as np
>>> np.linalg.norm(wv['Illinois'] - wv['Illini'])
3.3653798
>>> cos_similarity = np.dot(wv['Illinois'], wv['Illini']) / (
...     np.linalg.norm(wv['Illinois']) *\
...     np.linalg.norm(wv['Illini']))
>>> cos_similarity
0.5501352
>>> 1 - cos_similarity
0.4498648
~~~

> * word “Illini” refers to a group of people, usually football players and fans, rather than a single geographic region like “Illinois”
> * these distances mean that the words “Illini” and “Illinois” are only moderately close to one another in meaning.

#### 6.2.8.2 Plot US cities from a word2vec embedding on a 2D map of meaning


~~~python
>>> from nlpia.data.loaders import get_data
>>> cities = get_data('cities')
>>> cities.head(1).T
geonameid                       3039154
name                          El Tarter
asciiname                     El Tarter
alternatenames     Ehl Tarter,?? ??????
latitude                        42.5795
longitude                       1.65362
feature_class                         P
feature_code                        PPL
country_code                         AD
cc2                                 NaN
admin1_code                          02
admin2_code                         NaN
admin3_code                         NaN
admin4_code                         NaN
population                         1052
elevation                           NaN
dem                                1721
timezone                 Europe/Andorra
modification_date            2012-11-03

>>> us = cities[(cities.country_code == 'US') &\
...     (cities.admin1_code.notnull())].copy()
>>> states = pd.read_csv(\
...     'http://www.fonz.net/blog/wp-content/uploads/2008/04/states.csv')
>>> states = dict(zip(states.Abbreviation, states.State))
>>> us['city'] = us.name.copy()
>>> us['st'] = us.admin1_code.copy()
>>> us['state'] = us.st.map(states)
>>> us[us.columns[-3:]].head()
                     city  st    state
geonameid
4046255       Bay Minette  AL  Alabama
4046274              Edna  TX    Texas
4046319    Bayou La Batre  AL  Alabama
4046332         Henderson  TX    Texas
4046430           Natalia  TX    Texas

>>> vocab = pd.np.concatenate([us.city, us.st, us.state])
>>> vocab = np.array([word for word in vocab if word in wv.wv])
>>> vocab[:5]
array(['Edna', 'Henderson', 'Natalia', 'Yorktown', 'Brighton'])
~~~

> load data set and focus on columns of `state name` and `abbreviation` of each city 

~~~python
>>> city_plus_state = []
>>> for c, state, st in zip(us.city, us.state, us.st):
...     if c not in vocab:
...         continue
...     row = []
...     if state in vocab:
...         row.extend(wv[c] + wv[state])
...     else:
...         row.extend(wv[c] + wv[st])
...     city_plus_state.append(row)
>>> us_300D = pd.DataFrame(city_plus_state)
~~~

> add the `word vector of state name` with `word vector of city name` to deal with same city-name problem (a lot of large cities with the same name, like `Portland, Oregon` and `Portland, Maine`)

~~~python
>>> word_model.distance('man', 'nurse')
0.7453
>>> word_model.distance('woman', 'nurse')
0.5586
~~~

> word relationships
> 
> * depends on the corpus, it can represent different attributes (such as geographical proximity or cultural or economic similarities) 
> * also word vectors are biased	

~~~python
>>> from sklearn.decomposition import PCA
>>> pca = PCA(n_components=2)
>>> us_300D = get_data('cities_us_wordvectors')
>>> us_2D = pca.fit_transform(us_300D.iloc[:, :300])
>>> 
>>> import seaborn
>>> from matplotlib import pyplot as plt
>>> from nlpia.plots import offline_plotly_scatter_bubble
>>> df = get_data('cities_us_wordvectors_pca2_meta')
>>> html = offline_plotly_scatter_bubble(
...     df.sort_values('population', ascending=False)[:350].copy()\
...         .sort_values('population'),
...     filename='plotly_scatter_bubble.html',
...     x='x', y='y',
...     size_col='population', text_col='name', category_col='timezone',
...     xscale=None, yscale=None,  # 'log' or None
...     layout={}, marker={'sizeref': 3000})
{'sizemode': 'area', 'sizeref': 3000}
~~~

> using `PCA` to project these `city + state + abbrev vectors` into `2D plot`and the result is as below
>  
> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig08_alt.jpg)

### 6.2.9. Unnatural words

`word embedding` is not only for English words but also 

> * works for any sequence of symbols 
> * works for pictorial languages such as traditional Chinese and Japanese (Kanji) or the mysterious hieroglyphics in Egyptian tombs
> * works for unnatural words or ID numbers such as college course numbers (CS-101), model numbers (Koala E7270 or Galaga Pro), and even serial numbers, phone numbers, and ZIP codes

example: "[A non-NLP application of Word2Vec – Towards Data Science](https://medium.com/towards-data-science/a-non-nlp-application-of-word2vec-c637e35d3668)"

### 6.2.10. Document similarity with Doc2vec

<b>doc2vec</b>: 

> extend the Word2vec concept to sentences, paragraphs, or entire documents, during the training, the prediction
>
> * not only considers the previous words
> * but also the vector representing the paragraph or the document (as an additional word input to the prediction) 
> 
> over time, the algorithm learns a document or paragraph representation from the training set
>
> ![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/06fig10_alt.jpg)
>
> reference: "[Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053v2.pdf)"

how to train a doc2vec model

~~~python
>>> import multiprocessing
>>> num_cores = multiprocessing.cpu_count()
 
>>> from gensim.models.doc2vec import TaggedDocument, Doc2Vec  
>>> from gensim.utils import simple_preprocess
>>> corpus = ['This is the first document ...',\
...           'another document ...']
>>> training_corpus = []
>>> for i, text in enumerate(corpus):
...     tagged_doc = TaggedDocument(simple_preprocess(text), [i])
...     training_corpus.append(tagged_doc)
>>> model = Doc2Vec(size=100, min_count=2, workers=num_cores, iter=10)
>>> model.build_vocab(training_corpus)
>>> model.train(training_corpus, total_examples=model.corpus_count, epochs=model.iter)
~~~

> once the Doc2vec model is trained, you can infer document vectors for new, unseen documents by calling infer_vector on the instantiated and trained model

~~~python 
>>> model.infer_vector(simple_preprocess(\
...     'This is a completely unseen document'), steps=10)
~~~

> with these few steps, you can 
>
> * quickly train an entire corpus of documents and find similar documents (cosine distance)  
> * cluster the document vectors of a corpus with something like k-means to create a document classifier

## Summary

> * You’ve learned how word vectors and vector-oriented reasoning can solve some surprisingly subtle problems like analogy questions and nonsynonomy relationships between words.
> * You can now train Word2vec and other word vector embeddings on the words you use in your applications so that your NLP pipeline isn’t “polluted” by the GoogleNews meaning of words inherent in most Word2vec pretrained models.
> * You used gensim to explore, visualize, and even build your own word vector vocabularies.
> * A PCA projection of geographic word vectors like US city names can reveal the cultural closeness of places that are geographically far apart.
> * If you respect sentence boundaries with your n-grams and are efficient at setting up word pairs for training, you can greatly improve the accuracy of your latent semantic analysis word embeddings (see chapter 4)


