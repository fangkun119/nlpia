# Ch 3. Math with words (TF-IDF vectors)

> 3 increasingly powerful ways to represent words and their importance in a document

> * `Bags of words`: Vectors of word counts or frequencies
> * `Bags of n-grams`: Counts of word pairs (bigrams), triplets (trigrams), and so on
> * `TF-IDF vectors`: Word scores that better represent their importance

code: 

> * [ch03.md](../src/nlpia/book/examples/ch03.md)
> * [ch03.py](../src/nlpia/book/examples/ch03.py)
> * [ch03-2.py](../src/nlpia/book/examples/ch03-2.py)
> * [ch03_2.ipynb](../src/nlpia/book/examples/ch03_2.ipynb)
> * [ch03_zipf.py](../src/nlpia/book/examples/ch03_zipf.py)
> * [ch03_bm25.py](../src/nlpia/book/examples/ch03_bm25.py)

## 3.1. Bag of words

`Binary BOW` (Chapter 02) : 

> one-hot encoding of each word and then combined all those vectors with a binary OR (or clipped sum) to create a vector representation of a text

`Counted BOW` (`TF Vector`)

> counts the number of occurrences, or frequency, of each word in the given tex

~~~python
>>> from nltk.tokenize import TreebankWordTokenizer
>>> sentence = """The faster Harry got to the store, the faster Harry,
...     the faster, would get home."""
>>> tokenizer = TreebankWordTokenizer()
>>> tokens = tokenizer.tokenize(sentence.lower())
>>> tokens
['the', 'faster', 'harry', ..., 'would', 'get', 'home', '.']
>>> from collections import Counter
>>> bag_of_words = Counter(tokens)
>>> bag_of_words
Counter({'the': 4, 'faster': 3, 'harry': 2, ..., ’would': 1, 'get': 1, 'home': 1, '.': 1})
~~~

> notice: cannot rely on the order of your tokens (keys) in a Counter, it is differed with the implementation versions

> it is sufficient to do some powerful things such as detect spam, compute sentiment (positivity, happiness, and so on), and even detect subtle intent, like sarcasm

~~~python
>>> bag_of_words.most_common(4)
[('the', 4), (',', 3), ('faster', 3), ('harry', 2)]
~~~

`term frequency` (`TF`): 

> the number of times a word occurs in a given document, </br>
> also might be normalized (divided) by the number of terms in the document </br> 

~~~python
>>> times_harry_appears = bag_of_words['harry']
>>> num_unique_words = len(bag_of_words)
>>> tf = times_harry_appears / num_unique_words
>>> round(tf, 4)
0.1818
~~~

`normalize` for `calculate tf-vector and word frequency`: 

> instead of raw word counts to describe your documents in a corpus, you can use normalized term frequencies (by divided with the number of terms in the document)
> 
> example: </br>
> TF(“dog,” documentA) = 3/30 = .1 </br>
> TF(“dog,” documentB) = 100/580000 = .00017
> 


`stop word` for `calculate tf-vector and word frequency`: 

> “the” and the punctuation “,” aren’t very informative and should be ignored, along with a list of standard English stop words and punctuation 

<b>example</b>: introduction of kite in wikipedia

> text: [kite.txt](./resource/kite.txt)</br>
>
> notice: use `spaCy` in your production app rather than the NLTK components we used for these simple examples, because it does sentence segmentation and tokenization introduced in `chapter 11`, reference be found in [“spaCy 101: Everything you need to know”](https://spacy.io/usage/spacy-101#annotations-token).

> without using `stop word`: 

~~~python
>>> from collections import Counter
>>> from nltk.tokenize import TreebankWordTokenizer
>>> tokenizer = TreebankWordTokenizer()
>>> from nlpia.data.loaders import kite_text
>>> tokens = tokenizer.tokenize(kite_text.lower())
>>> token_counts = Counter(tokens)
>>> token_counts
Counter({'the': 26, 'a': 20, 'kite': 16, ',': 15, ...})
~~~

> using `stop word`: the vector make sense much more

~~~python
>>> import nltk
>>> nltk.download('stopwords', quiet=True)
True
>>> stopwords = nltk.corpus.stopwords.words('english')
>>> tokens = [x for x in tokens if x not in stopwords]
>>> kite_counts = Counter(tokens)
>>> kite_counts
Counter({'kite': 16,
         'traditionally': 1,
         'tethered': 2,
         'heavier-than-air': 1,
         'craft': 2,
         'wing': 5,
         'surfaces': 1,
         'react': 1,
         'air': 2,
         ...,
         'made': 1})}
~~~

## 3.2. Vectorizing

convert `kite_counts` above into `document vector`

~~~python
>>> document_vector = []
>>> doc_length = len(tokens)
>>> for key, value in kite_counts.most_common():
...     document_vector.append(value / doc_length)
>>> document_vector
[0.07207207207207207,
 0.06756756756756757,
 0.036036036036036036,
 ...,
 0.0045045045045045045]
~~~

but when considering many `document-vectors` for many `documents`

> all vectors should relative to something consistent before doing math on them in next step, they need to represent a position in a common space, the approach is as below: 
>
> 1. `step 1`: normalize the counts by calculating normalized term frequency instead of raw count in the document (as you did in the last section)
> 2. `step 2`: make all the vectors of standard length or dimension

code: 

> (1) build `lexicon`: the collections of words in your vocabulary which covers all documents

~~~python
>>> docs = []
>>> docs.append("The faster Harry got to the store, the faster and faster Harry would get home.")
>>> docs.append("Harry is hairy and faster than Jill.")
>>> docs.append("Jill is not as hairy as Harry.")
>>> 
>>> 
>>> doc_tokens = []
>>> for doc in docs:
...     doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
>>> len(doc_tokens[0])
17
>>> all_doc_tokens = sum(doc_tokens, [])
>>> len(all_doc_tokens)
33
>>> lexicon = sorted(set(all_doc_tokens))
>>> len(lexicon)
18
>>> lexicon
[',', '.', 'and', 'as', 'faster', 'get', 'got', 'hairy', 'harry', 'home', 'is', 'jill', 'not', 'store', 'than', 'the', 'to', 'would']
~~~

> (2) generate `document vector` for each documents based on the `lexion`

~~~python
>>> from collections import OrderedDict
>>> zero_vector = OrderedDict((token, 0) for token in lexicon)
>>> zero_vector
OrderedDict([(',', 0), ('.', 0), ('and', 0), ('as', 0), ('faster', 0), ('get', 0), ('got', 0), ('hairy', 0), ('harry', 0), ('home', 0), ('is', 0), ('jill', 0), ('not', 0), (‘store', 0), ('than', 0), ('the', 0), ('to', 0), ('would', 0)])

>>> import copy
>>> doc_vectors = []
>>> for doc in docs:
...     vec = copy.copy(zero_vector)
...     tokens = tokenizer.tokenize(doc.lower())
...     token_counts = Counter(tokens)
...     for key, value in token_counts.items():
...         vec[key] = value / len(lexicon)
...     doc_vectors.append(vec)
~~~

> reference about `linear algebra` and `vectors`: [Appendix 03](https://livebook.manning.com/book/natural-language-processing-in-action/appendix-c/app03)

### 3.2.1. Vector spaces

example of 2D Vector Space: 

![2d vector space example](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/03fig02_alt.jpg)

Curse of Dimensionality: [https://en.wikipedia.org/wiki/Curse_of_dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

* [Exploring Hyperspace](https://docs.google.com/presentation/d/1SEU8VL0KWPDKKZnBSaMxUBDDwI8yqIxu9RQtq2bpnNg)
* [Python annoy package](https://github.com/spotify/annoy) 
* [high dimensional approximate nearest neighbors](https://scholar.google.com/scholar?q=high+dimensional+approximate+nearest+neighbor)

NLP document vector space: 

> dimensionality of your vector space is the count of the number of distinct words that appear in the entire corpus

compare 2 NLP Document Vectors: 

> * `2-norm distance` (`Euclidean distance`)
> * `Cosine similarity` (`A⋅B = |A||B| * cos Θ`): do not care about the document length, just estimate of document similarity to find use of the same words about the same number of times in similar proportions (not similar use count) 

![cosine similarity](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/03fig03_alt.jpg)

`cosine similarity` 

> (1) is efficient to calculate (because based on `dot product`) like below as well as the code to compute it

~~~python
a.dot(b) == np.linalg.norm(a) * np.linalg.norm(b) / np.cos(theta)
~~~

~~~python
>>> import math
>>> def cosine_sim(vec1, vec2):
...     """ Let's convert our dictionaries to lists for easier matching."""
...     vec1 = [val for val in vec1.values()]
...     vec2 = [val for val in vec2.values()]
...
...     dot_prod = 0
...     for i, v in enumerate(vec1):
...         dot_prod += v * vec2[i]
...
...     mag_1 = math.sqrt(sum([x**2 for x in vec1]))
...     mag_2 = math.sqrt(sum([x**2 for x in vec2]))
...
...     return dot_prod / (mag_1 * mag_2)
~~~

> (2) has a and convenient range for most machine learning problems: -1 to +1 </br>
> 
> * close to `1`: means documents are using similar words in similar proportion
> * close to `0`: means documents share no words in common (but doesn’t means they are sure to have different meaning or topic) 
> * close to `-1`: won’t happen for `term-count vector` because term count just can’t be negative and seldom be seen in `TF vector`

> but in the next chapter, we develop a concept of words and topics that are “opposite” to each other. And this will show up as documents, words, and topics that have cosine similarities of less than zero, or even -1 

## 3.3. Zipf’s Law

`Zipf’s law` states that given some corpus of natural language utterances, the frequency of any word is inversely proportional to its rank in the frequency table (`Zipf定律`表示语料库中词的词频的log，与其在词频表中的Rank的log成反比). 

> * code similuation: [ch03_zipf.py](../src/nlpia/book/examples/ch03_zipf.py)
> * The first item in the ranked list will appear twice as often as the second, and three times as often as the third, ... 
> * It can also applied to many other things ([https://www.nature.com/articles/srep00812](https://www.nature.com/articles/srep00812)), such as population dynamics, economic output, and resource distribution

example: 

![zipf law](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/03fig04_alt.jpg)

> [https://github.com/totalgood/nlpia/blob/master/src/nlpia/book/examples/ch03_zipf.py](https://github.com/totalgood/nlpia/blob/master/src/nlpia/book/examples/ch03_zipf.py)

## 3.4. Topic modeling

why need `tf-idf`

> word counts are useful, but pure word count, even when normalized by the length of the document, doesn’t tell you much about the importance of that word in that document relative to the rest of the documents in the corpus

`IDF` (`Inverse document frequency`) 

> * is your window through Zipf in topic analysis
> * 2 ways of `term frequency counter`: (1) per document; (2) across the entire corpus


Example: 

(1) create a corpus with 2 documents

> doc 1: [kite\_intro\_doc.txt](./resource/kite.txt)</br>
> doc 2: [kite\_historical\_doc.txt](./resource/kite_hst.txt)</br>

(2) calculating `TF`

total word count in each document: 

~~~python
>>> from nlpia.data.loaders import kite_text, kite_history
>>> kite_intro = kite_text.lower()
>>> intro_tokens = tokenizer.tokenize(kite_intro)
>>> kite_history = kite_history.lower()
>>> history_tokens = tokenizer.tokenize(kite_history)
>>> intro_total = len(intro_tokens)
>>> intro_total    # doc 1: kite_intro_doc.txt’s term count
363
>>> history_total = len(history_tokens)
>>> history_total  # doc 2: kite_historical_doc.txt’s term count
297
~~~

term frequency of “kite” in each document: 

~~~python
>>> intro_tf = {}
>>> history_tf = {}
>>> intro_counts = Counter(intro_tokens)
>>> history_counts = Counter(history_tokens)
>>> intro_tf['kite']    = intro_counts['kite']    / intro_total   # tf[‘kite’] for doc 1: 0.0441
>>> history_tf['kite']  = history_counts['kite']  / history_total # tf[‘kite’] for doc 2: 0.0202
>>> intro_tf['and']     = intro_counts['and']     / intro_total   # tf[‘and’] for doc 1: 0.0275
>>> history_tf['and']   = history_counts['and']   / history_total # tf[‘and’] for doc 2: 0.0303
>>> intro_tf['china']   = intro_counts['china']   / intro_total   # tf[‘china’] for doc 1 
>>> history_tf['china'] = history_counts['china'] / history_total # tf[‘china’] for doc 2
~~~

term ‘add’ have close `tf` in 2 docs, but `kite` not , which means it is truely that `kite` have different `tf` in these 2 docs

(3) Calculate `IDF` 

> if a term appears in one document a lot of times, but occurs rarely in the rest of the corpus, one could assume it’s important to that document specifically.

`IDF`: the ratio of the total number of documents to the number of documents the term appears in, for example: 

> * 2 total documents / 2 documents contain “and”   = 2/2 = 1 
> * 2 total documents / 2 documents contain “kite”  = 2/2 = 1
> * 2 total documents / 1 document contains “China” = 2/1 = 2

`code`:

~~~python
>>> # documents containing the desired term
>>> num_docs_containing_and = 0
>>> for doc in [intro_tokens, history_tokens]:
...     if 'and' in doc:
...         num_docs_containing_and += 1
>>> num_docs_containing_kite = 0
>>> for doc in [intro_tokens, history_tokens]:
...     if 'and' in doc:
...         num_docs_containing_kite += 1
>>> num_docs_containing_china = 0
>>> for doc in [intro_tokens, history_tokens]:
...     if 'and' in doc:
...         num_docs_containing_china += 1
>>> # raw idf
>>> num_docs = 2
>>> intro_idf = {}
>>> raw_history_idf = {}
>>> raw_intro_idf['and']      = num_docs / num_docs_containing_and
>>> raw_history_idf['and']    = num_docs / num_docs_containing_and
>>> raw_intro_idf['kite']     = num_docs / num_docs_containing_kite
>>> raw_history_idf['kite']   = num_docs / num_docs_containing_kite
>>> raw_intro_idf['china']    = num_docs / num_docs_containing_china
>>> raw_history_idf['china']  = num_docs / num_docs_containing_china
~~~

(4) Calculate `TF-IDF`

~~~python
>>> #raw tf-idf for doc 1
>>> raw_intro_tfidf = {}
>>> raw_intro_tfidf['and']    = intro_tf['and'] * intro_idf['and']
>>> raw_intro_tfidf['kite']   = intro_tf['kite'] * intro_idf['kite']
>>> raw_intro_tfidf['china']  = intro_tf['china'] * intro_idf['china']
>>> #raw tf-idf for doc 2
>>> raw_history_tfidf = {}
>>> raw_history_tfidf['and']   = history_tf['and'] * history_idf['and']
>>> raw_history_tfidf['kite']  = history_tf['kite'] * history_idf['kite']
>>> raw_history_tfidf['china'] = history_tf['china'] * history_idf['china']
~~~

### 3.4.1. Return of Zipf

> understanding `tf-idf` in context of `zipf` distribution

example : two similar words `cat` and `dog`

> suppose document count is 1000000, and only 1 doc has term ‘cat’, 10 docs have term ‘dog’</br>
> * `raw idf` for `cat` is : 1000000 / 1  = `1000000` </br>
> * `raw idf` for `dog` is : 1000000 / 10 = `100000`  </br>
> it’s a very large difference on raw `idf`

like “cat” and “dog”

> even if they occur a similar number of times, the more frequent word will have an exponentially higher frequency than the less frequent one

so Zipf’s Law give below suggestions to ensures words such as “cat” and “dog” have similar counts

> scale all your word frequencies (and document frequencies) with the `log()` function, the inverse of `exp()`</br>
> reference: [https://ecommons.cornell.edu/bitstream/handle/1813/6721/87-881.pdf](https://ecommons.cornell.edu/bitstream/handle/1813/6721/87-881.pdf)

let’s apply the `log(raw_idf`) to previouse example </br>

> * cat_idf = log(1,000,000/1)  = 6
> * dog_idf = log(1,000,000/10) = 5

`tf-idf` after adding `log` on `raw_idf`: 

![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0090-01_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0090-01_alt.jpg)

sometimes, `log` will also be added into `tf`, so than `division` can become `substraction` 

~~~python
>>> log_tf = log(term_occurences_in_doc)  - log(num_terms_in_doc)
>>> log_log_idf = log(log(total_num_docs) - log(num_docs_containing_term))
>>> log_tf_idf = log_tf + log_idf
~~~

### 3.4.2. Relevance ranking

`TF-IDF` vectors for documents in corpus

~~~python
>>> document_tfidf_vectors = []
>>> for doc in docs:
...     vec = copy.copy(zero_vector)
...     tokens = tokenizer.tokenize(doc.lower())
...     token_counts = Counter(tokens)
...
...     for key, value in token_counts.items():
...         docs_containing_key = 0
...         for _doc in docs:
...             if key in _doc:
...                 docs_containing_key += 1
...         tf = value / len(lexicon)
...         if docs_containing_key:
...             idf = len(docs) / docs_containing_key
...         else:
...             idf = 0
...         vec[key] = tf * idf
...     document_tfidf_vectors.append(vec)
~~~ 

> in this implementation, keys that weren’t found in the lexicon are dropped to avoid a divide-by-zero error </br>
> another approach is using `[additive smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)` (`Laplace smoothing`): to +1 the denominator of every IDF calculation </br>


Two vectors are considered similar if their cosine similarity is high

![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0091-01.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/f0091-01.jpg)

query by `tf-idf similarity`

~~~python
>>> query = "How long does it take to get to the store?"
>>> query_vec = copy.copy(zero_vector)
>>> query_vec = copy.copy(zero_vector)

>>> tokens = tokenizer.tokenize(query.lower())
>>> token_counts = Counter(tokens)
 
>>> for key, value in token_counts.items():
...     docs_containing_key = 0
...     for _doc in documents:
...       if key in _doc.lower():
...         docs_containing_key += 1
...     if docs_containing_key == 0:
...         continue
...     tf = value / len(tokens)
...     idf = len(documents) / docs_containing_key
...    query_vec[key] = tf * idf
>>> cosine_sim(query_vec, document_tfidf_vectors[0])
0.5235048549676834
>>> cosine_sim(query_vec, document_tfidf_vectors[1])
0.0
>>> cosine_sim(query_vec, document_tfidf_vectors[2])
0.0
~~~

engineering implementations: 

* `index scan`: `O(n)`<br/>
* `inverted index`: `O(1)`<br/>
	* Inverted index: [https://en.wikipedia.org/wiki/Inverted_index](https://en.wikipedia.org/wiki/Inverted_index)</br>
	* Whoosh [https://pypi.python.org/pypi/Whoosh](https://pypi.python.org/pypi/Whoosh)</br>
	* GitHub - Mplsbeb/whoosh: A fast pure-Python search engine” [https://github.com/Mplsbeb/whoosh](https://github.com/Mplsbeb/whoosh)</br>

for chat-bot: 

> * store your training data in pairs of questions (or statements) and appropriate responses
> * then use TF-IDF to search for a question (or statement) most like the user input text

### 3.4.3. Tools

> using `libs` to implement logic above 

setup environment: 

using [appendix A](https://livebook.manning.com/book/natural-language-processing-in-action/appendix-a/app01) to install all packages about this book or below command

~~~shell
pip install scipy
pip install sklearn
~~~

code: 

~~~python
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = docs
>>> vectorizer = TfidfVectorizer(min_df=1)
>>> model = vectorizer.fit_transform(corpus)
>>> print(model.todense().round(2))
[[0.16 0.   0.48 0.21 0.21 0.   0.25 0.21 0.   0.   0.   0.21 0.   0.64	0.21 0.21]
 [0.37 0.   0.37 0.   0.   0.37 0.29 0.   0.37 0.37 0.   0.   0.49 0.  0.   0.  ]
 [0.   0.75 0.   0.   0.   0.29 0.22 0.   0.29 0.29 0.38 0.   0.   0.  0.   0.  ]]
~~~

### 3.4.4. Alternatives

optimations based on `IDF` 

> reference: [Word Embeddings Past, Present and Future by Piero Molino at AI with the Best 2017](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-3/ch03fn017)

![https://dpzbhybb2pdcj.cloudfront.net/lane/HighResolutionFigures/table_3-1.png](https://dpzbhybb2pdcj.cloudfront.net/lane/HighResolutionFigures/table_3-1.png)

> choosing most one bring most relevant results from the `weighting scheme`s above</br>

### 3.4.5. Okapi BM25

one of the optimations above is `BM25` (or its most recent variant `BM25F`)</br>

improvement compared with `TF-IDF cosine similarity` includes: </br>

> 1. normalize and smooth the similarity
> 2. ignore duplicate terms in the query document
> 3. effectively clipping the term frequencies for the query vector at 1
> 4. the dot product for the cosine similarity isn’t normalized by the TF-IDF vector norms (number of terms in the document and the query), but rather by a nonlinear function of the document length itself

~~~python
# q: query
# d: documents
q_idf * dot(q_tf, d_tf[i]) * 1.5 / 
 (dot(q_tf, d_tf[i]) + .25 + .75 * d_num_words[i] / d_num_words.mean()))
~~~

example: 

~~~python

from collections import Counter

import pandas as pd
from seaborn import plt
from mpl_toolkits.mplot3d import Axes3D

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction import TfidfVectorizer

CORPUS = ['"Hello world!"', 'Go fly a kite.', 'Kite World', 'Take a flying leap!', 'Should I fly home?']


def tfidf_corpus(docs=CORPUS):
    """ Count the words in a corpus and return a TfidfVectorizer() as well as all the TFIDF vecgtors for the corpus

    Args:
      docs (iterable of strs): a sequence of documents (strings)

    Returns:
      (TfidfVectorizer, tfidf_vectors)
    """
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(docs)
    return vectorizer, vectorizer.transform(docs)


def BM25Score(query_str, vectorizer, tfidfs, k1=1.5, b=0.75):
    query_tfidf = vectorizer.transform([query_str])[0]
    scores = []

    for idx, doc in enumerate(self.DocTF) :
        commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
        tmp_score = []
        doc_terms_len = self.DocLen[idx]
        for term in commonTerms :
            upper = (doc[term] * (k1+1))
            below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
            tmp_score.append(self.DocIDF[term] * upper / below)
        scores.append(sum(tmp_score))
    return scores
~~~

### 3.4.6. What’s next

> if your corpus isn’t too large, you might consider forging ahead with us into even more useful and accurate representations of the meaning of words and documents

> next chapter will introduce `semantic search`
>
> * it is much better than anything TF-IDF weighting and stemming and lemmatization can ever hope to achieve
> * just semantic word and topic vectors don’t scale to billions of documents
