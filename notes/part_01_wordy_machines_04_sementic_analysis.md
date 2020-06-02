# Chapter 4. Finding meaning in word counts (semantic analysis)

## code of this chapter

> LSA
>
> * [ch04_catdog_lsa_3x6x16.py](../src/nlpia/book/examples/ch04_catdog_lsa_3x6x16.py)
> * [ch04_catdog_lsa_4x8x200.py](../src/nlpia/book/examples/ch04_catdog_lsa_4x8x200.py)
> * [ch04_catdog_lsa_9x4x12.py](../src/nlpia/book/examples/ch04_catdog_lsa_9x4x12.py)
> * [ch04_catdog_lsa_sorted.py](../src/nlpia/book/examples/ch04_catdog_lsa_sorted.py)
> * [ch04_stanford_lsa.py](../src/nlpia/book/examples/ch04_stanford_lsa.py)
> 
> LDA
>
> * [ch04_spam_lda.py](../src/nlpia/book/examples/ch04_spam_lda.py)
> > PCA
> 
> * [ch04_sklearn_pca_source.py](../src/nlpia/book/examples/ch04_sklearn_pca_source.py)
>
> SVD 
> 
> * [ch04.svdfail.py](../src/nlpia/book/examples/ch04.svdfail.py)
> 
> examples of this chapter
> 
> * [ch04.py](../src/nlpia/book/examples/ch04.py)
> 
> others
> 
> * [ch4_fix_accuracy_discrepancy.py](../src/nlpia/book/examples/ch4_fix_accuracy_discrepancy.py)
> * [ch04_horse.py](../src/nlpia/book/examples/ch04_horse.py)
> * [ch04_ldia_comparison.py](../src/nlpia/book/examples/ch04_ldia_comparison.py)

## Introduction

<b>`tf-idf vectors`</b>: can be used to represent importance of `words`, `word sequences`, `n-grams` in a chunk of text</br>
<b>`latent semantic analysis (LSA)`</b>: represent the meaning of entire documents (not only the meaning of words as vectors), also can provide someone with the most meaningful words for a document

> * use weighted frequency scores from TF-IDF vectors to compute the topic “scores”
> * use correlation of normalized term frequencies with each other to group words together in topics

## 4.1. From word counts to topic scores

### 4.1.1. TF-IDF vectors and lemmatization

> <b>normalization approaches</b> such as `stemming`, `lemmatization` (词型还原，如cars->car, ate->eat) but can not find synonyms and sometimes erroneously lump together antonyms, words with opposite meaning</br>
> <b>edit-distance calculations</b> are better for identifying similarly spelled (or misspelled) words</br>

the limations are: 

* fail to find phrases with same meaning, such as `latent semantic indexing` and `latent semantic analysis` (see google n-gram viewer: [https://mng.bz/7Jnm](https://mng.bz/7Jnm))

* tell `frequency of word` uses in the documents (`whose vectors you combined or differenced`), but doesn’t tell meaning behand words 

	> `word-to-word tf-idf vector` (`word co-occurrence` or `correlation vectors`, by multiplying your TF-IDF matrix by itself) is an approach but too sparse

### 4.1.2. Topic vectors

`topic vectors` means 

> we need more informations such as `word signify`, ` combination of words`, need a vector that’s like a TF-IDF vector, but `more compact and more meaningful`

using the `topic vectors`: 

> * choose the dimension number according to the requirement
> * sums and differences mean a lot more than `tf-idf vectors`
> * distances between topic vectors is useful
> * don’t need to reprocess the entire corpus when comes a new document (except some algorithms such as `Latent Dirichlet allocation`)

challenges: 

> * polysemy (多义词）
> 	* Homonyms—Words with the same spelling and pronunciation, but different meanings
> 	* Zeugma—Use of two meanings of a word simultaneously in the same sentence
>	* Homonyms—Words with the same spelling and pronunciation, but different meanings
>	* Zeugma—Use of two meanings of a word simultaneously in the same sentence

### 4.1.3. Thought experiment

> It’s just a thought experiment (not real algorithm or implementation) for understanding the principle of `topic model`</br>
> 
> let’s assume 3 topics (`petness`,`animalness`, `cityness`) are generated with below approaches (the coropus has 6 topic words in total): 
> 
> * `word frequencies related topics` are added up
> * `weighted` these words like `tf-idf`
> * `word frequencies opposited to the topics` are substracted 

let’s think through how a human might decide mathematically which topics and words are connected

~~~python
>>> topic = {}
>>> tfidf = dict(list(zip('cat dog apple lion NYC love'.split(),
...     np.random.rand(6))))
>>> topic['petness'] = (
...		+0.3 * tfidf['cat']   +\	# general related with `petness`
...		+0.3 * tfidf['dog']   +\	# general related
...		 0.0 * tfidf['apple'] +\	# un-related
...		 0.0 * tfidf['lion']  -\	# un-related
...		-0.2 * tfidf['NYC']   +\	# negative related
...		+0.2 * tfidf['love'])		# weak related
>>> topic['animalness']  = (
...		+0.1 * tfidf['cat']   +\	# weak related with `animalness`
...		+0.1 * tfidf['dog']   +\	# weak related
...		-0.1 * tfidf['apple'] +\	# negative related
...		+0.5 * tfidf['lion']  +\	# strong related
...		+0.1 * tfidf['NYC']   +\	# weak related
...   	-0.1 * tfidf['love']) 		# negative related
>>> topic['cityness']    = ( 
...		+0.0 * tfidf['cat']   +\	# un-related with `cityness`
...		-0.1 * tfidf['dog']   +\	# negative related
...		+0.2 * tfidf['apple'] +\	# weak related (big-apple means New York)
...		-0.1 * tfidf['lion']  +\	# negative related
...		+0.5 * tfidf['NYC']   +\	# strong related
...		+0.1 * tfidf['love'])		# weak related
~~~

> * for each topic, have different weight on each words, which makes a `topic-vector`
> * for each document, it have a `tf-idf` vector for topic words 
>
> multiply a `topic-vector` and a `tf-idf` vecotr, we can get how does a document closed to a topic 

let’s flip the matrix in python code above to check how each word is related to the topics

~~~python
>>> word_vector = {}
>>> word_vector['cat']  =  +0.3 * topic['petness']    +\
...                        +0.1 * topic['animalness'] +\
...                        +0.0 * topic['cityness']
>>> word_vector['dog']  =  +0.3 * topic['petness']    +\
...                        +0.1 * topic['animalness'] +\
...                        -0.1 * topic['cityness']
>>> word_vector['apple']=  +0.0 * topic['petness']    +\
...                        -0.1 * topic['animalness'] +\
...                        +0.2 * topic['cityness']
>>> word_vector['lion'] =  +0.0 * topic['petness']    +\
...                        +0.5 * topic['animalness'] +\
...                        -0.1 * topic['cityness']
>>> word_vector['NYC']  =  -0.2 * topic['petness']    +\
...                        +0.1 * topic['animalness'] +\
...                        +0.5 * topic['cityness']
>>> word_vector['love'] =  +0.2 * topic['petness']    +\
...                        -0.1 * topic['animalness'] +\
...                        +0.1 * topic['cityness']
~~~

if regarding these 3 topics (`petness`,`animalness`, `cityness`) as 3 dimensions, below is the `word` in the topic space 

> ![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/04fig01_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/04fig01_alt.jpg)

related endevour: 

> * code commen sense above into algorithm by human labering: [https://www.wired.com/2016/03/doug-lenat-artificial-intelligence-common-sense-engine](https://www.wired.com/2016/03/doug-lenat-artificial-intelligence-common-sense-engine) 
> * use `SVD` to  identify patterns that represent the meanings of both the topics and the words: [https://upload.wikimedia.org/wikipedia/commons/7/70/Topic_model_scheme.webm#t=00:00:01,00:00:17.600](ttps://upload.wikimedia.org/wikipedia/commons/7/70/Topic_model_scheme.webm#t=00:00:01,00:00:17.600)

> still need a algorithm to transform `tf-idf vector` to `topic vector` 

### 4.1.4. An algorithm for scoring topics

estimate what a word or morpheme[6] signifie 

> * `morpheme`(词素) is the smallest meaningful parts of a word
> * for example, how do you tell `the “company” of a word` (陪伴，还是公司）?

most straight forward way is to `count co-occurrences in the same document`, implementations are: 

<b>Latent semantic analysis (`LSA`): also named as `latent semantic indexing (LSI)`</b>

> * works on table of `TF-IDF` vectors (also works on `BOW` vectors but not effective than `TF-IDF` vectors)
> * it can reduces the number of dimensions; (`PCA` reduce dimensions of number vectors, but `LSA` works on `TF-IDF` vectors) 
> 
> it will be introduced in `4.2`, belows are two cousins of `LSA`

`Cousin 1`: <b>Linear discriminant analysis (`LDA`)</b>

> * breaks down a document into only one topic
> * because is one dimensional, it doesn’t require singular value decomposition(SVD)
> * just compute the centroid (average or mean) of all your TF-IDF vectors for each side of a binary class (like spam and nonspam), the further a TF-IDF vector is along that line (the dot product of the TF-IDF vector with that line) tells you how close you are to one class or another

`Cousin 2`: <b>Latent Dirichlet allocation (`LDiA`)</b>: [https://ppasupat.github.io/a9online/1140.html#latent-dirichlet-allocation-lda-](https://ppasupat.github.io/a9online/1140.html#latent-dirichlet-allocation-lda-)

> * LDiA takes the math of LSA in a different direction (nonlinear statistical algorithm) and generally takes much longer to train
> 	* Weakness: the long training time makes LDiA less practical for many real-world applications, and it should rarely be the first approach you try
> 	* Stregnth: topics created sometimes more closely mirror human intuition about words and topics, and more easier to be explained
> * used to generate vectors that capture the semantics of a word or document (more like LSA because it can break down documents into as many topics as you like) 
> 	* useful for some single-document problems such as document summarization
>	* corpus becomes the document, document becomes sentences, this is how `gensim` and other packages use LDiA to identify the most “central” sentences of a document, and create a machine-generated summary (reference: [https://livebook.manning.com/book/natural-language-processing-in-action/chapter-4/ch04fn011](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-4/ch04fn011)) 
> 	* thib book generated some of the text in the “About this book” section using similar math, but implemented in a neural network (see chapter 12)

### 4.1.5. LDA (Linear discriminant analysis) classifier

very old algorithm which is not flashy but very effective

> * LDA is one of the most straightforward and fast dimension reduction and classification models you’ll find. But this book may be one of the only places you’ll read about it, because it’s not very flashy.[8] But in many applications, you’ll find it has much better accuracy than the fancier state-of-the art algorithms published in the latest paper
> * papaer back to 1990s, [Automatic Document Classification Natural Language Processing Statistical Analysis and Expert System Techniques used together](https://www.researchgate.net/profile/Georges_Hebrail/publication/221299406_Automatic_Document_Classification_Natural_Language_Processing_Statistical_Analysis_and_Expert_System_Techniques_used_together/links/0c960516cf4968b29e000000.pdf)

implementation can be found in machine learning or NLP libs such as `sklearn`, but aslo a very simplified one is provided as below, which only has 3 steps: 

> * average position (centroid) of all the TF-IDF vectors within the class (such as spam SMS messages)
> * average position (centroid) of all the TF-IDF vectors not in the class (such as nonspam SMS messages)
> * vector difference between the centroids (the line that connects them)

code is as below: 

step (1) load text messages: 638 of the total 4837 SMS are spam

~~~python
>>> import pandas as pd
>>> from nlpia.data.loaders import get_data
>>> pd.options.display.width = 120
>>> sms = get_data('sms-spam')
>>> index = ['sms{}{}'.format(i, '!'*j) for (i,j) in\
...     zip(range(len(sms)), sms.spam)]
>>> sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
>>> sms['spam'] = sms.spam.astype(int)
>>> len(sms)
4837
>>> sms.spam.sum()
638
>>> sms.head(6)
      spam                                               text
sms0     0  Go until jurong point, crazy.. Available only ...
sms1     0                      Ok lar... Joking wif u oni...
sms2!    1  Free entry in 2 a wkly comp to win FA Cup fina...
sms3     0  U dun say so early hor... U c already then say...
sms4     0  Nah I don't think he goes to usf, he lives aro...
sms5!    1  FreeMsg Hey there darling it's been 3 week's n...
~~~	

step (2) tokenization and generate TF-IDF vector 

~~~python
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs  = tfidf_model.fit_transform(\
...     raw_documents=sms.text).toarray()
>>> tfidf_docs.shape
(4837, 9232) # 4837 sms, 9232 words in vocalubary
>>> sms.spam.sum()
638
~~~

> * 4837 sms, 9232 words in vocalubary (2 times); 638 spam sms, 9232 words (10 times)

> * <b>we need `semantic analysis`</b>

> 	because ususaly `Naive Bayes classifier` won’t work well when your vocabulary is much larger (10 times in our example) than the number of labeled examples 

step (3) simplest semantic analysis technique, LDA 

> * use `sklearn.discriminant_analysis.LinearDiscriminant-Analysis`
> * or just compute the centroids of your binary class as below 

~~~python
>>> mask = sms.spam.astype(bool).values
>>> spam_centroid = tfidf_docs[mask].mean(axis=0)
>>> ham_centroid = tfidf_docs[~mask].mean(axis=0)
 
>>> spam_centroid.round(2)
array([0.06, 0.  , 0.  , ..., 0.  , 0.  , 0.  ])
>>> ham_centroid.round(2)
array([0.02, 0.01, 0.  , ..., 0.  , 0.  , 0.  ])
~~~

step (4) subtract one centroid from the other to get the line between them

~~~python
>>> spamminess_score = tfidf_docs.dot(spam_centroid -\
...     ham_centroid)
>>> spamminess_score.round(2)
array([-0.01, -0.02,  0.04, ..., -0.01, -0.  ,  0.  ])
~~~

step (5) plot the tf-idf vectors

![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/04fig02_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/04fig02_alt.jpg)

> the arrow from the nonspam centroid to the spam centroid is the line that defines your trained model

step (6) normalize the score to range between 0 and 1

~~~python
>>> from sklearn.preprocessing import MinMaxScaler
>>> sms['lda_score'] = MinMaxScaler().fit_transform(\
...     spamminess_score.reshape(-1,1))
>>> sms['lda_predict'] = (sms.lda_score > .5).astype(int)
>>> sms['spam lda_predict lda_score'.split()].round(2).head(6)
       spam  lda_predict  lda_score
sms0      0            0       0.23
sms1      0            0       0.18
sms2!     1            1       0.72
sms3      0            0       0.18
sms4      0            0       0.29
sms5!     1            1       0.55
~~~

step (7) precision of the model

~~~python
>>> (1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms)).round(3)
0.977
~~~

> alghtough this is not a `testing score` (just a `training score`).  But LDA is a very simple model, with few parameters, so it should generalize well, as long as <b>your SMS messages are representative of the messages you intend to classify</b>.  ( for `cross validate`, refer to [`Appendix 4`](https://livebook.manning.com/book/natural-language-processing-in-action/appendix-d/app04) )

> * `strength`: semantic analysis doesn’t rely on individual words.[9] Semantic analysis gathers up words with similar semantics (such as spamminess) and uses them all together
> * `weekness`: this training set has a `limited vocabulary` and some non-English words in it. So your test messages need to use similar words if you want them to be classified correctly

> Actually, a Naive Bayes classifier and a logistic regression model are both equivalent to this simple LDA model. 

step (8) confusion matrix

~~~python
>>> from pugnlp.stats import Confusion
>>> Confusion(sms['spam lda_predict'.split()])
lda_predict     0    1
spam
0            4135   64
1              45  593
~~~

> it looks nice. You could adjust the 0.5 threshold on your score if the false positives (64) or false negatives (45) were out of balance

## 4.2. Latent semantic analysis

### 4.2.0 `LSA` and `SVD`

SVD（奇异值分解）: 
 
> * break down your TF-IDF term-document matrix into 3 simpler matrices, and then they can be multiplied back together to produce the original matrix. 
> * you can truncate those matrices (ignore some rows and columns) before multiplying them back together, which reduces the number of dimensions, whichis also called `truncated singular value decomposition` (`truncated SVD`).
> * Visualizations and explanations: [https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf#chapter.16](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf#chapter.16)

When using `SVD` this way in `NLP`, you call it latent semantic analysis (`LSA`). Besides, you can also uses `dimension rotation` tricks to improve the accuracy: 

> * `LSA` uses `SVD` to find the combinations of words that are responsible, together, for the biggest variation in the data
> * You can rotate your `TF-IDF vectors` so that the new dimensions (basis vectors) of your rotated vectors all align with these maximum variance directions
> * Each dimensions (axes) becomes a combination of word frequencies rather than a single word frequency)

> dimension rotation is also appliable to `PCA` (`PCA` is also based on `truncated SVD` but applied on images or other high-dimensional data, like time series

LSA compresses more meaning into fewer dimensions. We only have to retain the high-variance dimensions

> * LSA give you the high varianced dimensions (topics)
> * You can discard low-variance dimension (topics), which are usually distractions, noise
> * Is has similar effect like the “IDF” part of TF-IDF and `stop-words`, but it is better and optimal, never discard words, only discard topics

### 4.2.1. Your thought experiment made real

> for a small corpus of short documents (such as tweets, chat messages, and lines of poetry) it takes only a few dimensions (topics) to capture the semantics of those documents

> this chapter is just for understanding `LSA` algorithm, reference is [Learning Low-Dimensional Metrics](https://papers.nips.cc/paper/7002-learning-low-dimensional-metrics.pdf)

`Listing 4.2`. Topic-word matrix for `LSA` on `16 short sentences` about `cats, dogs, and NYC`

~~~python
>>> from nlpia.book.examples.ch04_catdog_lsa_3x6x16\
...     import word_topic_vectors
>>> word_topic_vectors.T.round(1)
      cat  dog  apple  lion  nyc  love   # 6 columns for “topic vectors” of 6 words
top0 -0.6 -0.4    0.5  -0.3  0.4  -0.1   # topic 1: highest variance dimension
top1 -0.1 -0.3   -0.4  -0.1  0.1   0.8	 # topic 2: 2nd variance dimension
top2 -0.3  0.8   -0.1  -0.5  0.0   0.1	 # topic 3: animal
~~~

<b>topic 0 dimension (</b> `highest variance` axis <b>)</b>

> axis direction is from `non-city` to `city`: 
> 
> * `cat`,`dog`,`lion` (weight < 0) -> `love` (weight close to 0) -> `nyc`, `apple` (wight > 0)
> 
> comments: 
> 
> * `nyc` is also called `big apple` 
> * `several sentences` mentions about “nyc” and “apple” 
> * `several sentences` don’t use those words at all

<b>topic 1 dimension (</b> `2nd variance` axos <b>)</b>

> axis direction is from `non-love` to `love`: 
> 
> * `apple`,`dog`, `apple`, ... to `love`
> 
> comments: 

> `2nd Variance` dimension, it found “love” was a more important topic than “animalness” 

<b>topic 2 dimension (</b> `3rd variance` <b>)</b>: 

> axis direnction is from `lion` to `dog`,`love`
> 
> * it is the dimension about “dog” mixed with a little bit of “love”
> * it also shows “cat” is relegated to the “anti-cityness” topic (negative cityness), because cats and cities aren’t mentioned together much 

<b>Mad Libs Game</b>: [https://en.wikipedia.org/wiki/Mad_Libs](https://en.wikipedia.org/wiki/Mad_Libs)

>  replace a word in a sentence with a foreign word, or even a made-up word. Then ask a friend to guess what that word means. It won’t be too far off from a valid translation.

> for example: “`Awas`! `Awas`! Tom is behind you! Run!” (`Awas` means “danger” or “watch out” in Indonesian)

<b>shorter documents</b>, like sentences, are better for `LSA` than large documents such as articles or books? 

> * `LSA` is suitable for short document. It is because meaning of the words in same sentance are usually closely related, but far apart in a longer document
> * unlike `LSA`, `word2vec` could tighten up the meaning of word vectors, because the context is tightened up even further

## 4.3. Singular value decomposition

> `document-term matrix`, each row is a vecotor for the `BOW` of a sentence, 11 sentences in total

~~~python
>>> from nlpia.book.examples.ch04_catdog_lsa_sorted\
...     import lsa_models, prettify_tdm
>>> bow_svd, tfidf_svd = lsa_models() # bow: bag-of-words
>>> prettify_tdm(**bow_svd) # tdm: term-docmument-matrix (6 terms in this corpus); 
   cat dog apple lion nyc love
text
0            1        1                                 NYC is the Big Apple.
1            1        1                        NYC is known as the Big Apple.
2                     1    1                                      I love NYC!
3            1        1           I wore a hat to the Big Apple party in NYC.
4            1        1                       Come to NYC. See the Big Apple!
5            1                             Manhattan is called the Big Apple.
6    1                                New York is a big city for a small cat.
7    1            1           The lion, a big cat, is the king of the jungle.
8    1                     1                               I love my pet cat.
9                     1    1                      I love New York City (NYC).
10   1   1                                            Your dog chased mycat.
~~~

> `term-document matrix` (the transpose of the document-term matrix)

~~~python
>>> tdm = bow_svd['tdm']    # tdm: term-docmument-matrix
>>> tdm
        0   1   2   3   4   5   6   7   8   9   10
cat     0   0   0   0   0   0   1   1   1   0    1
dog     0   0   0   0   0   0   0   0   0   0    1
apple   1   1   0   1   1   1   0   0   0   0    0
lion    0   0   0   0   0   0   0   1   0   0    0
nyc     1   1   1   1   1   0   0   0   0   1    0
love    0   0   1   0   0   0   0   0   1   1    0
~~~

> we will firstly use `SVD` on the `term-document matrix` (not only on `BOW`, `SVD` also works on `tf-idf` matrix) 

<b>SVD</b>: decomposing any matrix into 3 “factors”,  

> W<sub>mxn</sub> = U<sub>mxp</sub> . S<sub>pxp</sub> . V<sub>pxn</sub><sup>T</sup>

> * `m` is the `number of terms`     in the vocabular
> * `n` is the `number of documents` in the corpus
> * `p` is the `number of topics`    in the corpus

> run `SVD` on a `BOW` (or `TF-IDF`) `term-document matrix`, will find combinations of words that belong together, this done by simulataneously finding below 2 things: 

> * correlation of term use between documents
> * correlation of documents with each other

> and `linear combinations of terms` that have the greatest variation across the corpus, which will become the `topics`

> it also gives you the linear transformation (rotation) of your term-document vectors to convert those vectors into shorter topic vectors for each document

Below chapters is about how to use `SVD` to reduce the dimension 

### 4.3.1. `U` — `left singular vectors`

> `m`: `number of terms`; `n`: `number of documents`; `p`:`number of topics`

W<sub>mxn</sub> = <b>`U`<sub>`m`x`p`</sub></b> . S<sub>pxp</sub> . V<sub>pxn</sub><sup>T</sup>

> * `U`, the `term-topic matrix`,  is the cross-correlation between `words and topics` based on `word co-occurrence in the same document`</br>
> * `U` is a square matrix (`p` = `m`) until you start truncating it (deleting columns, `p` < `m`)

~~~python
|>>> import numpy as np
|>>> # tdm: term-topic matrix of the corpus
|>>> #      notice that PCA or some other model lib require `document-term` vetor rather than `term-document` vecotor, thus the `tdm` (`term-document-matrix`) need to be `transposed` before passed to the `fit` function
|>>> U, s, Vt = np.linalg.svd(tdm) 
|>>> import pandas as pd
|>>> pd.DataFrame(U, index=tdm.index).round(2)
|>>> # U: topic-document-matrix
|>>> #    each cell is about how important each word is to each topic
|>>> # 
          0     1     2     3     4     5
cat   -0.04  0.83 -0.38 -0.00  0.11 -0.38
dog   -0.00  0.21 -0.18 -0.71 -0.39  0.52
apple -0.62 -0.21 -0.51  0.00  0.49  0.27
lion  -0.00  0.21 -0.18  0.71 -0.39  0.52
nyc   -0.75 -0.00  0.24 -0.00 -0.52 -0.32
love  -0.22  0.42  0.69  0.00  0.41  0.37
~~~

> * input: `tdm` - `term-document-matrix` from `BOW` or `TF-IDF` </br>
> * ouput: `U`<sub>`m`x`p`</sub>  - `term-topic-martrix`, each cell is about how important each word is to each topic</br>
> 
> the `term-topic-martrix`(`U`) multiplis a `word-document column vector` will get an `topic-document column vector`

### 4.3.2. S — singular values

> `S`<sub>`p`x`p`</sub>:  
> 
> * is a diagonal matrix
> * it tells how much information is captured by each dimension in your new semantic (topic) vector space

~~~python
>>> s.round(1)
array([3.1, 2.2, 1.8, 1. , 0.8, 0.5]) # numpy use array to store `s` for saving space
>>> S = np.zeros((len(U), len(Vt)))
>>> pd.np.fill_diagonal(S, s) # convert the array back to matrix
>>> pd.DataFrame(S).round(1)
    0    1    2    3    4    5    6    7    8    9    10
0  3.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
1  0.0  2.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
2  0.0  0.0  1.8  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
3  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
4  0.0  0.0  0.0  0.0  0.8  0.0  0.0  0.0  0.0  0.0  0.0
5  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0
~~~

> * `p` = 6 here, it is the topic number before reduced the dimensionality by truncating it’s diagonal elements </br>
> * rest columns (column 6 - 10) filled with 0 are just for matrix multiply</br>
> 
> for using SVD, you can: 
> 
> * start zeroing out the dimensions at the lower right and work your way up and to the left
> * stop  zeroing out these singular values when the error in your topic model starts to contribute significantly to the overall NLP pipeline error

### 4.3.3. V<sup>T</sup> — right singular vectors

> V<sup>T</sup> (right singular vectors): is the document-document matrix, measures how often documents use the same topics 

~~~python
>>> pd.DataFrame(Vt).round(2)
      0     1     2     3     4     5     6     7     8     9     10
0  -0.44 -0.44 -0.31 -0.44 -0.44 -0.20 -0.01 -0.01 -0.08 -0.31 -0.01
1  -0.09 -0.09  0.19 -0.09 -0.09 -0.09  0.37  0.47  0.56  0.19  0.47
2  -0.16 -0.16  0.52 -0.16 -0.16 -0.29 -0.22 -0.32  0.17  0.52 -0.32
3   0.00 -0.00 -0.00  0.00  0.00  0.00 -0.00  0.71  0.00 -0.00 -0.71
4  -0.04 -0.04 -0.14 -0.04 -0.04  0.58  0.13 -0.33  0.62 -0.14 -0.33
5  -0.09 -0.09  0.10 -0.09 -0.09  0.51 -0.73  0.27 -0.01  0.10  0.27
6  -0.57  0.21  0.11  0.33 -0.31  0.34  0.34 -0.00 -0.34  0.23  0.00
7  -0.32  0.47  0.25 -0.63  0.41  0.07  0.07  0.00 -0.07 -0.18  0.00
8  -0.50  0.29 -0.20  0.41  0.16 -0.37 -0.37 -0.00  0.37 -0.17  0.00
9  -0.15 -0.15 -0.59 -0.15  0.42  0.04  0.04 -0.00 -0.04  0.63 -0.00
10 -0.26 -0.62  0.33  0.24  0.54  0.09  0.09 -0.00 -0.09 -0.23 -0.00
~~~

### 4.3.4. SVD matrix orientation

> for models with `sklearn` package (`Naive Bayes sentiment model` in charptor 2 and chaptor 3 `TF-IDF vectors` chaptor 3)
> 
> * training set is created as a `document-term matrix` (`row` corresponds to document, `column` corresponds to `word` or `feature`) as required by `sklearn`
> 
> for `SVD` (chaptor 4) 
> 
> * need input as `term-document matrix` as required by SVD linear algebra

### 4.3.5. Truncating the topics

> currently without reducing deminsons: you just created some new words and called them “topics” because they each combine words together in various ratios

<b>How many topics will be enough</b> to capture the essence of a document?

> one way to measure the accuracy of LSA is to see how accurately you can recreate a term-document matrix from a topic-document matrix, below is a re-creation example 

~~~python
>>> err = []
>>> for numdim in range(len(s), 0, -1):
...     S[numdim - 1, numdim - 1] = 0
...     reconstructed_tdm = U.dot(S).dot(Vt)
...     err.append(np.sqrt(((\
...         reconstructed_tdm - tdm).values.flatten() ** 2).sum()
...         / np.product(tdm.shape)))
>>> np.array(err).round(2)
array([0.06, 0.12, 0.17, 0.28, 0.39, 0.55])
~~~

> the more you truncate, the more the error grows, below is the plot

![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/04fig03_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/04fig03_alt.jpg)

> 	TF-IDF	 vectors will perform slightly better if you plan to retain only a few topics in your model

> in some cases you may find that you get perfect accuracy, after eliminating several of the dimensions in your term-document matrix
> 
> 	* this is because SVD algorithm behind LSA “notices” if words are always used together and puts them together in a topic
> 
> even if you don’t plan to use a topic model in your pipeline, LSA (SVD) 
> 
>  * can be a great way to compress your word-document matrices
>  * identify potential compound words or n-grams for your pipeline

## 4.4. Principal component analysis

`PCA`

> `PCA` is another name for SVD for dimension reduction, the `PCA model` in `scikit-learn` as some tweaks to the `SVD` to improve the accuracy of NLP pipeline as below

> * `sklearn.PCA` automatically `“centers” your data` by subtracting off the mean word frequencies
> * `flip_sign` funciton is used to deterministically compute the sign of the singular vectors 
> 	* reference code : nlpia.book.examples.ch04\_sklearn\_pca\_source)
> * `“whitening” step` (optionally) similarly with ignoring the singular values in `SVD` but not just simply setting all the singular values in `S` to one, it divides the data by these variances (like the `sklearn.StandardScaler` transform does)
> 	* reference : [“Deep Learning Tutorial - PCA and Whitening”](http://mccormickml.com/2014/06/03/deep-learning-tutorial-pca-and-whitening/)

Run LSA on large data set:

> * If you have a huge corpus and you urgently need topic vectors (LSA), skip to chapter 13 and check out [gensim.models.LsiModel](https://radimrehurek.com/gensim/models/lsimodel.html) 
> * RocketML’s parallelization of the SVD algorithm: [http://rocketml.net](http://rocketml.net)

### 4.4.1. PCA on 3D vectors

> Like `PCA` on 3D vectors as below, when `SVD` applied on `word vectors`, it preserves the structure, information content, of your vectors by maximizing the variance along the dimensions of your lower-dimensional “shadow” of the high-dimensional space. 

~~~python
>>> import pandas as pd
>>> pd.set_option('display.max_columns', 6)
>>> from sklearn.decomposition import PCA
>>> import seaborn
>>> from matplotlib import pyplot as plt
>>> from nlpia.data.loaders import get_data
 
>>> df = get_data('pointcloud').sample(1000)
>>> pca = PCA(n_components=2)
>>> df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))
>>> df2d.plot(kind='scatter', x='x', y='y')
>>> plt.show()
~~~

> in above example, the direction of the figures might be flipped, but the max variance axis will be the x axis, the 2nd variance one is y axis; 

> another example is `horse_plot.py` script in the `nlpia/data` to cast a 3D-horse to a 2D diagram

![https://dpzbhybb2pdcj.cloudfront.net/lane/HighResolutionFigures/figure_4-5.png](https://dpzbhybb2pdcj.cloudfront.net/lane/HighResolutionFigures/figure_4-5.png) 

> the left-side one is from a nonlinear transformation, the other one is the embedding algorithm will be introduced in chaptor 6

### 4.4.2. Stop horsing around and get back to NLP

find the `principal components` using `SVD` on the 5,000 SMS messages labeled as `spam (or not)`

> * the `vocabulary and variety` of topics discussed is relatively small 
> 	* it is a limited set of SMS messages from a university
> 	* thus set the topics number to a small value 16
> * use both `scikit-learn PCA model` and `truncated SVD model` to see the difference

TruncatedSVD

> * designed for sparse matrices, which is always exists in `BOW` or `TF-IDF` matrix

sklearn PCA model 

> * faster than `TruncatedSVD` but consume much more memory by using `dense matrix`

code

> load sms message

~~~python
>>> import pandas as pd
>>> from nlpia.data.loaders import get_data
>>> pd.options.display.width = 120
 
>>> sms = get_data('sms-spam')
>>> index = ['sms{}{}'.format(i, '!'*j) 
 for (i,j) in zip(range(len(sms)), sms.spam)]
>>> sms.index = index
>>> sms.head(6)
 
       spam                                               text
sms0      0  Go until jurong point, crazy.. Available only ...
sms1      0                      Ok lar... Joking wif u oni...
sms2!     1  Free entry in 2 a wkly comp to win FA Cup fina...
sms3      0  U dun say so early hor... U c already then say...
sms4      0  Nah I don't think he goes to usf, he lives aro...
sms5!     1  FreeMsg Hey there darling it's been 3 week's n...
~~~

> calculate `tf-idf`

~~~python
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
 
>>> tfidf = TfidfVectorizer(tokenizer=casual_tokenize) # tokenizer is casual_tokenize
>>> tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
>>> len(tfidf.vocabulary_)
9232
 
>>> tfidf_docs = pd.DataFrame(tfidf_docs)
>>> tfidf_docs = tfidf_docs - tfidf_docs.mean()
>>> tfidf_docs.shape
(4837, 9232) # 4,837 SMS messages, 9,232 different 1-gram tokens (vocabular), larger than message number
>>> sms.spam.sum()
638 # 638 (13%) messages are spam, it’s an unbalanced training set (1:8)
~~~

> since the larger number of tokens compared with message number, as well as the un-balanced data-set, this data-set will cause [overfitting](https://en.wikipedia.org/wiki/Overfitting)

> 





### 4.4.3. Using PCA for SMS message semantic analysis

### 4.4.4. Using truncated SVD for SMS message semantic analysis

### 4.4.5. How well does LSA work for spam classification?

## 4.5. Latent Dirichlet allocation (LDiA)

### 4.5.1. The LDiA idea

### 4.5.2. LDiA topic model for SMS messages

### 4.5.3. LDiA + LDA = spam classifier

### 4.5.4. A fairer comparison: 32 LDiA topics

## 4.6. Distance and similarity

## 4.7. Steering with feedback

### 4.7.1. Linear discriminant analysis

## 4.8. Topic vector power

### 4.8.1. Semantic search

### 4.8.2. Improvements

