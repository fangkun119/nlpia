# Part 1: wordy machines

# Chapter 01. Packets of thought (NLP overview)

## 1.1. Natural language vs. programming language

> programing following specific rules and the logic of the processing is pre-defined; but NLP application has no predifined logic, it is based on the data and model training


## 1.2. The magic

> huge amount of data

### 1.2.1. Machines that converse

> NLP can’t be directly translated into a precise set of mathematical operations, they extract information and instructions

### 1.2.2. The Math

> compare math (NLP) with human and the challenges </br>
> 1. context </br>
> [https://github.com/MathieuCliche/Sarcasm_detector](https://github.com/MathieuCliche/Sarcasm_detector)</br>
> [http://www.thesarcasmdetector.com/](http://www.thesarcasmdetector.com/)</br>
> 2. sementic analysis </br> 
> majorly based on statistical relationships minning</br>
> some gramma check can be down with finite state machine like regex</br>
> 3. decoding challenge</br> 
> `good morning` has nothing to do with `morning`</br>
> 4. do not have common sense knowledge about the world of human processor 

## 1.3. Practical applications

![table](https://dpzbhybb2pdcj.cloudfront.net/lane/HighResolutionFigures/table_1-1.png)

> project introduction and websites are linked in the book 

## 1.4. Language through a computer’s “eyes”

> what if we use `FST` (`finite state transducer`) based program to process natrual language </br>

### 1.4.1. The language of locks

> mechanical analogies to how regular expressions work </br>

### 1.4.2. Regular expressions

pattern-based engines that rely on `regular grammars` like `Amazon Alexa` and `Google Now` 

> * regex like `grep` aren’t true `regular grammar` which will cause "crash" (running forever) in some situations</br>
> * `regular grammars` are used to find sequences within a user statement that they know how to respond to in a chat-bot</br>

Natural Language: 

> * are not regular</br>
> * are not context free</br>
> * can't be defined by any formal grammar</br>

### 1.4.3. A simple chatbot

pattern-based chat-bot for understanding greeting

~~~python
import re
r = "(hi|hello|hey)[ ]*([a-z]*)"
re.match(r, 'Hello Rosa', flags=re.IGNORECASE)
# <_sre.SRE_Match object; span=(0, 10), match='Hello Rosa'>
re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE)
# <_sre.SRE_Match object; span=(0, 5), match='hi ho'>
re.match(r, "hey, what's up", flags=re.IGNORECASE)
# <_sre.SRE_Match object; span=(0, 3), match='hey>
~~~

a more complicated one

~~~python
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"\
	r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)
re_greeting.match('Hello Rosa')
# <_sre.SRE_Match object; span=(0, 10), match='Hello Rosa'>
re_greeting.match('Hello Rosa').groups()
# ('Hello', None, None, 'Rosa')
re_greeting.match("Good morning Rosa")
# <_sre.SRE_Match object; span=(0, 17), match="Good morning Rosa">
re_greeting.match("Good Manning Rosa")
re_greeting.match('Good evening Rosa Parks').groups()
# ('Good evening', 'Good ', 'evening', 'Rosa')
re_greeting.match("Good Morn'n Rosa")
# <_sre.SRE_Match object; span=(0, 16), match="Good Morn'n Rosa">
re_greeting.match("yo Rosa")
# <_sre.SRE_Match object; span=(0, 7), match='yo Rosa'>
~~~

more and more complicated

~~~python
my_names = set(['rosa', 'rose', 'chatty', 'chatbot', 'bot', 'chatterbot'])
curt_names = set(['hal', 'you', 'u'])
greeter_name = ''
match = re_greeting.match(input())
if match:
	at_name = match.groups()[-1]
	if at_name in curt_names:
		print("Good one.")
	elif at_name.lower() in my_names:
		print("Hi {}, How are you?".format(greeter_name))
~~~

### 1.4.4. Another way (statistical or machine learning approach)

`character sequence matches` like `Jaccard`, `Levenshtein` and `Euclidean vector distance`</br>

> * suitable for find spelling errors or typos</br>
> * fail to find sementically similarities 

vector representations of natural language words like `word2vec`
> * introduced in later chapter
 
`bag-of-word vectors`</br>

> * pipeline: tokenizer -> stop words filter -> rare words filter</br>
> * it generates sparse, high-dimensional vectors but good enough for application like spam filter </br>
> * usage example 1: find the combination of words most likely to follow a particular bag of words? </br> 
> * usage example 2: find the closest bag of words to some target </br>
> * curse of dimensionality: millions of dimensions for a 3-gram vocabulary computed from a large corpus

## 1.5. A brief overflight of hyperspace

> how to create an application from the vectors in hyperspace

## 1.6. Word order and grammar

<b>grammar</b>: goven the rules of word orders, which is discarded by bag of words or word vector</br>

sometimes word order does not matter like below: 

~~~python
>>> from itertools import permutations
 
>>> [" ".join(combo) for combo in\
...     permutations("Good morning Rosa!".split(), 3)]
['Good morning Rosa!',
 'Good Rosa! morning',
 'morning Good Rosa!',
 'morning Rosa! Good',
 'Rosa! Good morning',
 'Rosa! morning Good']
 
>>> # factorial(3) == 6 
~~~

but in many situations, it matters a lot

~~~python
>>> s = """Find textbooks with titles containing 'NLP',
...     or 'natural' and 'language', or
...     'computational' and  'linguistics'."""
>>> len(set(s.split()))
12
>>> import numpy as np
>>> # factorial(12) = arange(1, 13).prod() = 479001600
>>> np.arange(1, 12 + 1).prod()  
479001600
~~~

approach to use word order: 

> natural language syntax tree parsers</br>
> packages: `SyntaxNet` (Parsey McParseface) and `SpaCy`</br>
> evaluation ([https://spacy.io/docs/api/](https://spacy.io/docs/api/.)): accuracy of SpaCy (93%), SyntaxNet (94%), Stanford’s CoreNLP (90%)

## 1.7. A chatbot natural language pipeline

the pipeline is similar to the Q&A pipline named `Taming` introduced in [http://www.manning.com/books/taming-text](http://www.manning.com/books/taming-text)</br>

processing stages and algorithms used are as below as well as the figure: 

> * `parse`: extract features, structured numerical data, from natural language text
> * `analyze`: generate and combine features by scoring text for sentiment, grammaticality, and semantics
> * `generate`: Compose possible responses using templates, search, or language models
> * `execute`: Plan statements based on conversation history and objectives, and select the next response

![](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/01fig03_alt.jpg)

here is the comment for this figure: 

> * some chat-bot can be symplified according to it's functionality</br>
> * newly deep-learning or statiscal model might introduct more complicated piplines</br>
> * persist data for re-training in each steps </br>
> * feed-back loop for model iterations </br>
> * `text generation` step is a core feature in `chat-bot`, but not oftern not a part of  is a part of 

## 1.8. Processing in depth

![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/01fig04_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/01fig04_alt.jpg) 

4 stages generated by `SpaCY` pipeline

> * `part-of-speech tagging` (`POS tagging`) : for generating features with finite state transducer in `nltk.tag package`</br>
> * `Entity relationships` and `Knowledge base`: for populating information about a perticular domain 

chat-bot's decision can be made with these information but aslo can only based on these a few upper layers， e.g.: 

> * [ChatterBot](https://github.com/gunthercox/ChatterBot) only use dit distance” (Levenshtein distance) to match the input with the statement stored in DB </br>
> * [`Will Chat Bot`](https://github.com/skoczen/will) use `regular expressions` based programming to match then request and answers, which is suitable for Q&A system and task-execution assistant bots like Lex, Siri and Google Now </br> 

other techincals

> python `regex` package will replace `re` packege in the future </br>
> [TRE `agrep` or `approximate grep`](https://github.com/laurikari/tre) will replace UNIX `grep` in the future </br>
> `shallow NLP`: can do many thing but only little, if any, human supervision (labeling or curating of text) is required (chaptor 6) </br> 
> `Simple neural networks`:  can often be used for unsupervised feature extraction </br>


## 1.9. Natural language IQ

![https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/01fig05_alt.jpg](https://dpzbhybb2pdcj.cloudfront.net/lane/Figures/01fig05_alt.jpg)

details: [https://livebook.manning.com/book/natural-language-processing-in-action/chapter-1/186](https://livebook.manning.com/book/natural-language-processing-in-action/chapter-1/186)


