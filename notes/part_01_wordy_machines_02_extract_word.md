# Ch 02. Build your vocabulary (word tokenization)

> something considered when using NLP: </br>
> 
> * tokenize: need to process like tokens like `we’ll` or `""` (Chapter 1)</br>
> * stemming: regex based</br>
> * syllables, prefixes, suffixes like `re`, `ing`, `pre`</br>
> * morphemes like "apples" (s)</br>
> * invisible words like "You, do not do that" behands the word "Don't" (Chapter 4)</br>
> * seperate strings into word-phrase like recognize "ice cream" with `n-gram`</br>
> * estimate document frequency like `tf-idf` (Chapter 3)</br>
> * lossy feature extraction like `BOW` (`bag of word`) retains enough of the information for many model like sentiment analyzers or spam detection</br>

## 2.1. Challenges of Stemming

> many patterns like <i>`working->work`</i> is correct but <i>`sing->s`</i> is wrong</br>
> 
> * regex based `conventional stemming approaches` is introduced in this chapter<br> 
> * statistical clustering approaches (`semantic stems`,`lemmas`,`synonyms`) is introduced in `chapter 5`

## 2.2. Building your vocabulary with a tokenizer

> Concept from `compilers` are still available for `NLP tokenizer`, such as `scanner`(or `lexer`,`lexicon`), `context-free grammars` (`CFG`), `terminals` for pattern matching and information extration. more is introduced in `chapter 11` 

building blocks

|NLP    |Compiler|
|:-----|:------|
|tokenizer|canner, lexer, lexical analyzer|
|vocabulary|lexicon|
|parser|compiler|
|token, term, word, or n-gram|token, symbol, or terminal symbol|

tokenizing example:

```python
>>> sentence = "Thomas Jefferson began building Monticello at the age of 26."
>>> sentence.split()
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26.']
```
> python `split()` already done good job for this tokenizer, except the `26.` is error which should be `26` or [`26`, `[EOS]`]

encoding tokens of this sentence into `one-hot` vector

```python
>>> binary_vector = sorted(dict([(token, 1) for token in sentence.split()]).iteritems())
>>> 
>>> import pandas as pd
>>> sentence = "Thomas Jefferson began building Monticello at the age of 26."
>>> df = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()])), columns=['sent']).T
>>> df
      26.  Jefferson  Monticello  Thomas  age  at  began  building  of  the
sent    1          1           1       1    1   1      1         1   1    1
```

encoding more sentences 

```python
>>> sentences = """Thomas Jefferson began building Monticello at the\
...   age of 26.\n"""
>>> sentences += """Construction was done mostly by local masons and\
...   carpenters.\n"""
>>> sentences += "He moved into the South Pavilion in 1770.\n"
>>> sentences += """Turning Monticello into a neoclassical masterpiece\
...   was Jefferson's obsession."""
>>> corpus = {}
>>> for i, sent in enumerate(sentences.split('\n')):
...     corpus['sent{}'.format(i)] = dict((tok, 1) for tok in
...         sent.split())
>>> df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
>>> df[df.columns[:10]]
        1770.  26.  Construction   ...    Pavilion  South  Thomas
sent0      0    1             0   ...           0      0       1
sent1      0    0             1   ...           0      0       0
sent2      1    0             0   ...           1      1       0
sent3      0    0             0   ...           0      0       0
```

### 2.2.1. Dot product

<b>dot product</b> is used to check `similarities` between sentences by `counting the number of overlapping tokens`.

example and for understanding `dot product`

~~~python
>>> v1 = pd.np.array([1, 2, 3])
>>> v2 = pd.np.array([2, 3, 4])
>>> v1.dot(v2)
20
>>> (v1 * v2).sum()
20
>>> sum([x1 * x2 for x1, x2 in zip(v1, v2)])
20
~~~

`dot product` is also equivalent to the `matrix product`(
by `np.matmul()` or `@ operator`) as below

~~~python
# all vectors can be turned into Nx1 or 1xN matrices
v1.reshape(-1, 1).T @ v2.reshape(-1, 1)
~~~

### 2.2.2. Measuring bag-of-words overlap

continue the example in 2.2.1, find the `bow` overlap

~~~python
>>> df = df.T
>>> df.sent0.dot(df.sent1)
0
>>> df.sent0.dot(df.sent2)
1
>>> df.sent0.dot(df.sent3)
1
~~~

> notice that 0 does not means there is no overlap, it might just 2 different word with same meaning and can not be counted

find the word in overlap

~~~python
>>> [(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v]
[('Monticello', 1)]
~~~

<b>Euclidean distance</b> can also be applied on 2 `bow` vector to measure the `angle between these vectors`

<b>Addition</b>, <b>subtraction</b>, <b>OR</b>, <b>AND</b>, ... can aslo been applied on `bow` vector

### 2.2.3. A token improvement

> split the words not only by write-space, but aslo by `punctuation` such as `commas`, `periods`, `quotes`, `semicolons`, and even `hyphens (dashes)`

~~~python
>>> import re
>>> sentence = """Thomas Jefferson began building Monticello at the\
...   age of 26."""
>>> tokens = re.split(r'[-\s.,;!?]+', sentence)
>>> tokens
['Thomas',
 'Jefferson',
 'began',
 'building',
 'Monticello',
 'at',
 'the',
 'age',
 'of',
 '26',
 '']
~~~

more about `regex`:

> [https://livebook.manning.com/book/natural-language-processing-in-action/chapter-2/88](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-2/88)</br>
> [Appendix B: https://livebook.manning.com/book/natural-language-processing-in-action/appendix-b/app02](https://livebook.manning.com/book/natural-language-processing-in-action/appendix-b/app02)<br/>

some tips: 

> 1. `precompile` to increase the speed when regex number is larger than `MAXCACHE=100`</br>
> 2. `pip install regex` to install `regex` which will replace `re` package, which includes support for: (1) overlapping match sets
multithreading; (2) feature-complete support for unicode; (3) approximate regular expression matches (similar to TRE’s agrep on UNIX systems); (4) larger default MAXCACHE (500 regexes)

~~~python
>>> sentence = """Thomas Jefferson began building Monticello at the\
...   age of 26."""
>>> tokens = pattern.split(sentence)
>>> [x for x in tokens if x and x not in '- \t\n.,;!?']
['Thomas',
 'Jefferson',
 'began',
 'building',
 'Monticello',
 'at',
 'the',
 'age',
 'of',
 '26']
~~~

<b>more libs for tokenize: </b>

|name  |description                                                  |
|:-----|:------------------------------------------------------------|
|SpaCy |accurate , flexible, fast, Python                            |
|Stanford CoreNLP|more accurate, less flexible, fast, on Java 8      |
|NLTK | standard used by many NLP contests and comparisons, popular, Python|

Example 1 : NLTK RegexpTokenizer 

~~~python
>>> from nltk.tokenize import RegexpTokenizer
>>> tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
>>> tokenizer.tokenize(sentence)
['Thomas',
 'Jefferson',
 'began',
 'building',
 'Monticello',
 'at',
 'the',
 'age',
 'of',
 '26',
 '.']
 ~~~
 
 Example 2 : NLTK Treebank Word Tokenizer better than regex in English
 
 > [http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.treebank](http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.treebank) 
 
 ~~~python
 >>> from nltk.tokenize import TreebankWordTokenizer
>>> sentence = """Monticello wasn't designated as UNESCO World Heritage\
...   Site until 1987."""
>>> tokenizer = TreebankWordTokenizer()
>>> tokenizer.tokenize(sentence)
['Monticello',
 'was',
 "n't",
 'designated',
 'as',
 'UNESCO',
 'World',
 'Heritage',
 'Site',
 'until',
 '1987',
 '.']
 ~~~

Example 3: NLTK `casual_tokenize` for short, informal, emoticon-laced texts from social networks

~~~python
>>> from nltk.tokenize.casual import casual_tokenize
>>> message = """RT @TJMonticello Best day everrrrrrr at Monticello.\
...   Awesommmmmmeeeeeeee day :*)"""
>>> casual_tokenize(message)
['RT', '@TJMonticello',
 'Best', 'day','everrrrrrr', 'at', 'Monticello', '.',
 'Awesommmmmmeeeeeeee', 'day', ':*)']
>>> casual_tokenize(message, reduce_len=True, strip_handles=True)
['RT',
 'Best', 'day', 'everrr', 'at', 'Monticello', '.',
 'Awesommmeee', 'day', ':*)']
~~~

### 2.2.4. Extending your vocabulary with n-grams

<b>(1) n-grams</b></br>

> for mining phrases like “ice cream”, “beyond the pale”, “Johann Sebastian Bach” or “riddle me this.” , but `n-grams` merely have to be frequent enough together to catch the attention of your token counters

example: tokenize to 2-gram

~~~python
>>> tokenize_2grams("Thomas Jefferson began building Monticello at the\
...   age of 26.")
['Thomas Jefferson',
 'Jefferson began',
 'began building',
 'building Monticello',
 'Monticello at',
 'at the',
 'the age',
 'age of',
 'of 26']
~~~

example: build 2-gram, 3-gram from 1-gram tokenizer output

~~~python
>>> sentence = """Thomas Jefferson began building Monticello at the\
...   age of 26."""
>>> pattern = re.compile(r"([-\s.,;!?])+")
>>> tokens = pattern.split(sentence)
>>> tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
>>> tokens
['Thomas',
 'Jefferson',
 'began',
 'building',
 'Monticello',
 'at',
 'the',
 'age',
 'of',
 '26']

>>> from nltk.util import ngrams
>>> list(ngrams(tokens, 2))
[('Thomas', 'Jefferson'),
 ('Jefferson', 'began'),
 ('began', 'building'),
 ('building', 'Monticello'),
 ('Monticello', 'at'),
 ('at', 'the'),
 ('the', 'age'),
 ('age', 'of'),
 ('of', '26')]
>>> two_grams = list(ngrams(tokens, 2))
>>> [" ".join(x) for x in two_grams]
['Thomas Jefferson',
 'Jefferson began',
 'began building',
 'building Monticello',
 'Monticello at',
 'at the',
 'the age',
 'age of',
 'of 26'] 

>>> list(ngrams(tokens, 3))
[('Thomas', 'Jefferson', 'began'),
 ('Jefferson', 'began', 'building'),
 ('began', 'building', 'Monticello'),
 ('building', 'Monticello', 'at'),
 ('Monticello', 'at', 'the'),
 ('at', 'the', 'age'),
 ('the', 'age', 'of'),
 ('age', 'of', '26')]
~~~

> * `ngrams function` of the NLTK library returns a `Python generator`</br>
> * rare n-grams (like "of 26", "Jefferson begin") won’t be helpful for classification problems</br>
> * n-gram will increase vocabulary size exponentially. If feature vector dimensionality exceeds the length of all documents, the feature extraction step is counterproductive (in chapter 3 will use doc-frequency to filter them)</br>
> * over frequent grams like "at the" are also need to be ignored as a `stop word`, because they loses the utility for discriminating doc-meanings. If a token or n-gram occurs in more than 25% of all the documents in your corpus, you usually ignore it.

<b>(2) stop word</b></br>

> * `stop word` dict for NLTK's corpora: [https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip)</br>
> * filtering `stop word` might also lose import information sometimes, such as "to"/"as" in the 4-gram "reported to the CEO" and "reported as the CEO"</br>
> * filtering `stop word` do not have much contribution for reduce the dimensionality, thus we fiter them only for imporving accuracy not for processing ablity</br>

example 1: filtering a few stop-words with raw python

~~~python
>>> stop_words = ['a', 'an', 'the', 'on', 'of', 'off', 'this', 'is']
>>> tokens = ['the', 'house', 'is', 'on', 'fire']
>>> tokens_without_stopwords = [x for x in tokens if x not in stop_words]
>>> print(tokens_without_stopwords)
['house', 'fire']
~~~

example 2: stop-words from NLTK

~~~python
>>> import nltk
>>> nltk.download('stopwords')
>>> stop_words = nltk.corpus.stopwords.words('english')
>>> len(stop_words)
153
>>> stop_words[:7]
['i', 'me', 'my', 'myself', 'we', 'our', 'ours']
>>> # stop words oftenly occured when contractions are split and stemmed using NLTK tokenizers and stemmers
>>> [sw for sw in stopwords if len(sw) == 1]
['i', 'a', 's', 't', 'd', 'm', 'o', 'y']
~~~

example 3: merge sklearn stop-words with NLTK stop-words

~~~python
>>> from sklearn.feature_extraction.text import\
...   ENGLISH_STOP_WORDS as sklearn_stop_words
>>> len(sklearn_stop_words)
318
>>> len(stop_words)
179
>>> len(stop_words.union(sklearn_stop_words))
378
>>> len(stop_words.intersection(sklearn_stop_words))
119
~~~

### 2.2.5. Normalizing your vocabulary

> target: 
> 
> * tokens that mean similar things are combined into a single, normalized form</br>
> * prevent overfitting</br>
> * increase recall</br>

<b>(1) case folding</b></br>

> * consolidate multiple “spellings” of a word that differ only in their capitalization</br>
> * some words will lose information if doing case folding, such as "Doctor" (degree) and "doctor"</br>
> * case folding the whole document will also prevent advanced tokenizers that can split camel case words like “WordPerfect,” “FedEx,” or “stringVariableName.”</br>
> * a better approach for case normalization is to <b>lowercase only the first word of a sentence</b> and allow all other words to retain their capitalization, but also can not correctly process the sentence like "Joe Smith, the word smith, with a cup of joe"</br>
> * To avoid this potential loss of information, many NLP pipelines <b>don’t normalize for case</b> at all, but, for example, "The" at the start of a sentence can not be recognized as a stop word</br>
> * The best way is to try and measure several different approaches.

example:

~~~python
>>> tokens = ['House', 'Visitor', 'Center']
>>> # this is only for python 3
>>> # in python 2 there is bug that shift all ASICC number than words like "resumé" can not be correctly processed
>>> normalized_tokens = [x.lower() for x in tokens]
>>> print(normalized_tokens)
['house', 'visitor', 'center']
~~~

<b>(2) Stemming</b></br>

> * it could reduce demension like regarded "house”/“houses”/"housing" as 1 stem "hous"</br>
> * but in search engine, it also could greatly reduce the “precision” score by increasing "false-positive rate" (thus most search engines allow you to turn off stemming and even case normalization by putting quotes around a word or phrase)</br>
> * stemmers are only really used in large-scale information retrieval applications (keyword search)</br>

stem algorithms: (1) Porter Stemmer ([implementation](https://github.com/jedijulia/porter-stemmer/blob/master/stemmer.py)); (2) [Snowball Stemmer](http://snowball.tartarus.org/texts/introduction.html)

~~~python
>>> from nltk.stem.porter import PorterStemmer
>>> stemmer = PorterStemmer()
>>> # Porter Stemmer retains the trailing apostrophe(') 
>>> # to mark the words has been stemed, unless you strip it
>>> ' '.join([stemmer.stem(w).strip("'") for w in
...   "dish washer's washed dishes".split()])
'dish washer wash dish'
~~~

<b>(3) Lemmatization (词型还原）</b>

> chapter 12 will show how to reduce the complexity chatbot logic by lemmatization </br>
> 
> * pros: reduce demension, making the model more precise
> * cons: less precision, e.g.: bank (river) and banking (finance) are different 

`lemmatization` is potentially more accurate than `stemming` 

> * some lemmatizers use the word’s `part of speech (POS) tag` (`词性标注`) in addition to its spelling</br>
> * if you really want to use `stemmer`, it is better to use a lemmatizer right before the stemmer</br>

example: 

~~~python
>>> nltk.download('wordnet')
>>> from nltk.stem import WordNetLemmatizer
>>> lemmatizer = WordNetLemmatizer()
>>> # default pos is "n" (norn)
>>> lemmatizer.lemmatize("better") 
'better'
>>> lemmatizer.lemmatize("better", pos="a")
'good'
>>> lemmatizer.lemmatize("good", pos="a")
'good'
>>> lemmatizer.lemmatize("goods", pos="a")
'goods'
>>> lemmatizer.lemmatize("goods", pos="n")
'good'
>>> # won't errorly treat it as "good" like stemmer
>>> lemmatizer.lemmatize("goodness", pos="n")
'goodness'
>>> lemmatizer.lemmatize("best", pos="a")
'best'
~~~

<b>Use Cases</b>

> * NLP packages, such as `spaCy`, don’t provide stemming functions and only offer lemmatization methods</br>
> * `stemming`, `lemmatization`, and even `case folding` will increase recall but reduce the precision</br>
> * search engine always combine `stemmed` and `unstemmed` versions of words, and using ranking to banlance the precision and recall
> * besides above technique, `additional metadata` are also used 

important: try to avoid stemming and lemmatization unless you have a limited amount of text that contains usages and capitalizations of the words you are interested in.

## 2.3. Sentiment

### 2.3.1. VADER—A rule-based sentiment analyzer

> find keywords in the text and map each one to numerical scores or weights in a dictionary

VADER algorithm implementation: `nltk.sentiment.vader`, `vaderSentiment` (`pip install vaderSentiment`, and more detail is in [https://github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment))

~~~python
>>> from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
>>> sa = SentimentIntensityAnalyzer()
>>> sa.lexicon
{ ...
':(': -1.9,
':)': 2.0,
...
'pls': 0.3,
'plz': 0.3,
...
'great': 3.1,
... }
>>> [(tok, score) for tok, score in sa.lexicon.items()
...   if " " in tok]
[("( '}{' )", 1.6),
 ("can't stand", -2.0),
 ('fed up', -1.8),
 ('screwed up', -1.5)]
>>> sa.polarity_scores(text=\
...   "Python is very readable and it's great for NLP.")
{'compound': 0.6249, 'neg': 0.0, 'neu': 0.661,
'pos': 0.339}
>>> sa.polarity_scores(text=\
...   "Python is not a bad choice for most applications.")
{'compound': 0.431, 'neg': 0.0, 'neu': 0.711,
'pos': 0.289}
~~~

example

~~~python
>>> corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
...           "Horrible! Completely useless. :(",
...           "It was OK. Some good and some bad things."]
>>> for doc in corpus:
...     scores = sa.polarity_scores(doc)
...     print('{:+}: {}'.format(scores['compound'], doc))
+0.9428: Absolutely perfect! Love it! :-) :-) :-)
-0.8768: Horrible! Completely useless. :(
+0.3254: It was OK. Some good and some bad things.
~~~

### 2.3.2. Naive Bayes

> relies on a labeled set of statements or documents to train a machine learning model to create those rules (labels can come from user rating, user's hashtag, ...)




