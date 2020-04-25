# 00 Resource

## 1. Open Source Search Engine

* [http://duckduckgo.com](http://duckduckgo.com)</br>
* [http://gigablast.com/search?c=main&q=open+source+search+engine](http://gigablast.com/search?c=main&q=open+source+search+engine)</br>
* [https://www.qwant.com/web](https://www.qwant.com/web)</br>
* [https://en.wikipedia.org/wiki/Wikia_Search](https://en.wikipedia.org/wiki/Wikia_Search)</br>

## 2. NLP Application Examples

* [Guessing passwords from social network profiles](http://www.sciencemag.org/news/2017/09/artificial-intelligence-just-made-guessing-your-password-whole-lot-easier)</br>
* [Chatbot lawyer overturns 160,000 parking tickets in London and New York](www.theguardian.com/technology/2016/jun/28/chatbot-ai-lawyer-donotpay-parking-tickets-london-new-york)</br>
* [GitHub - craigboman/gutenberg: Librarian working with project gutenberg data, for NLP and machine learning purposes](https://github.com/craigboman/gutenberg)</br>
* [Longitudial Detection of Dementia Through Lexical and Syntactic Changes in Writing](ftp://ftp.cs.toronto.edu/dist/gh/Le-MSc-2010.pdf) — Masters thesis by Xuan Le on psychology diagnosis with NLP </br>
* [Time Series Matching: a Multi-filter Approach by Zhihua Wang](https://www.cs.nyu.edu/web/Research/Theses/wang_zhihua.pdf) — Songs, audio clips, and other time series can be discretized and searched with dynamic programming algorithms analogous to Levenshtein distance</br>
* [NELL, Never Ending Language Learning](http://rtw.ml.cmu.edu/rtw/publications) — CMU’s constantly evolving knowledge base that learns by scraping natural language text</br>
* [How the NSA identified Satoshi Nakamoto](https://medium.com/cryptomuse/how-the-nsa-caught-satoshi-nakamoto-868affcef595) — Wired Magazine and the NSA identified Satoshi Nakamoto using NLP, or stylometry</br>
* [Stylometry](https://en.wikipedia.org/wiki/Stylometry) and [Authorship Attribution for Social Media Forensics](http://www.parkjonghyuk.net/lecture/2017-2nd-lecture/forensic/s8.pdf) — Style/pattern matching and clustering of natural language text (also music and artwork) for authorship and attribution.
* [Online dictionaries like Your Dictionary ](http://examples.yourdictionary.com/) can be scraped for grammatically correct sentences with POS labels, which can be used to [train your own Parsey McParseface] (https://ai.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html) syntax tree and POS tagger
* [Identifying ‘Fake News’ with NLP](https://nycdatascience.com/blog/student-works/identifying-fake-news-nlp/) by Julia Goldstein and Mike Ghoul at NYC Data Science Academy.
* [simpleNumericalFactChecker ](https://github.com/uclmr/simpleNumericalFactChecker) by Andreas Vlachos ([git](https://github.com/andreasvlachos)) and information extraction (see chapter 11) could be used to rank publishers, authors, and reporters for truthfulness. Might be combined with Julia Goldstein’s “fake news” predictor.
* [The artificial-adversary ](https://github.com/airbnb/artificial-adversary) package by Jack Dai, an intern at Airbnb—Obfuscates natural language text (turning phrases like ‘you are great’ into ‘ur gr8’). You could train a machine learning classifier to detect and translate English into obfuscated English or [L33T](https://sites.google.com/site/inhainternetlanguage/different-internet-languages/l33t). You could also train a stemmer (an autoencoder with the obfuscator generating character features) to decipher obfuscated words so your NLP pipeline can handle obfuscated text without retraining. 

## 3. Courses and tutorials

* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) by David Jurafsky and James H. Martin

	> The next book you should read if you’re serious about NLP. Jurafsky and Martin are more thorough and rigorous in their explanation of NLP concepts. They have whole chapters on topics that we largely ignore, like finite state transducers (FSTs), hidden Markhov models (HMMs), part-of-speech (POS) tagging, syntactic parsing, discourse coherence, machine translation, summarization, and dialog systems.

* [MIT Artificial General Intelligence course 6.S099](https://agi.mit.edu) led by Lex Fridman Feb 2018

	> MIT’s free, interactive (public competition!) AGI course. It’s probably the most thorough and rigorous free course on artificial intelligence engineering you can find.

* [Textacy: NLP, before and after spaCy](https://github.com/chartbeat-labs/textacy)

	> Topic modeling wrapper for SpaCy.

* [MIT Natural Language and the Computer Representation of Knowledge course 6-863j lecture notes](http://mng.bz/vOdM) for Spring 2003.

* [Singular value decomposition (SVD)](http://people.revoledu.com/kardi/tutorial/LinearAlgebra/SVD.html) by Kardi Teknomo, PhD.

* [An Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf) by Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze.

## 4. Tools and packages

* [nlpia](http://github.com/totalgood/nlpia)

	> NLP datasets, tools, and example scripts from this book

* [OpenFST](http://openfst.org/twiki/bin/view/FST/WebHome) by Tom Bagby, Dan Bikel, Kyle Gorman, Mehryar Mohri et al

	> Open Source C++ Finite State Transducer implementation

* [pyfst](https://github.com/vchahun/pyfst) by Victor Chahuneau

	> A Python interface to OpenFST

* [Stanford CoreNLP—Natural language software](https://stanfordnlp.github.io/CoreNLP/) by Christopher D. Manning et al

	> Java library with state-of-the-art sentence segmentation, datetime extraction, POS tagging, grammar checker, and so on

* [stanford-corenlp 3.8.0](https://pypi.org/project/stanford-corenlp/)

	> Python interface to Stanford CoreNLP

* [keras](https://blog.keras.io/)

	> High-level API for constructing both Tensor-Flow and Theano computational graphs (neural nets)

## 5. Research papers and talks

> gain a deep understanding of a topic</br>
> try to repeat the experiments of researchers</br>
> modify them in some way</br>

### Vector space models and semantic search

* [Semantic Vector Encoding and Similarity Search Using Fulltext Search Engines](https://arxiv.org/pdf/1706.00957.pdf)

	> Jan Rygl et al. were able to use a conventional inverted index to implement efficient semantic search for all of Wikipedia.

* [Learning Low-Dimensional Metrics](https://papers.nips.cc/paper/7002-learning-low-dimensional-metrics.pdf)

	> Lalit Jain et al. were able to incorporate human judgement into pairwise distance metrics, which can be used for better decision-making and unsupervised clustering of word vectors and topic vectors. For example, recruiters can use this to steer a content-based recommendation engine that matches resumes with job descriptions.

* [RAND-WALK: A latent variable model approach to word embeddings](https://arxiv.org/pdf/1502.03520.pdf) by Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski

	> Explains the latest (2016) understanding of the “vector-oriented reasoning” of Word2vec and other word vector space models, particularly analogy questions

* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) by Tomas Mikolov, Greg Corrado, Kai Chen, and Jeffrey Dean at Google, Sep 2013

	> First publication of the Word2vec model, including an implementation in C++ and pretrained models using a Google News corpus

* [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean at Google

	> Describes refinements to the Word2vec model that improved its accuracy, including subsampling and negative sampling

* [From Distributional to Semantic Similarity](https://www.era.lib.ed.ac.uk/bitstream/handle/1842/563/IP030023.pdf) 2003 Ph.D. Thesis by James Richard Curran 

	> Lots of classic information retrieval (full-text search) research, including TF-IDF normalization and page rank techniques for web search

### Finance

* [Predicting Stock Returns by Automatically Analyzing Company News Announcements ](http://www.stagirit.org/sites/default/files/articles/a_0275_ssrn-id2684558.pdf)

	> Bella Dubrov used gensim’s Doc2vec to predict stock prices based on company announcements with excellent explanations of Word2vec and Doc2vec.

* [Building a Quantitative Trading Strategy to Beat the S&P 500](https://www.youtube.com/watch?v=ll6Tq-wTXXw)

	> At PyCon 2016, Karen Rubin explained how she discovered that female CEOs are predictive of rising stock prices, though not as strongly as she initially thought.

### Question answering systems

* [Keras-based LSTM/CNN models for Visual Question Answering](https://github.com/avisingh599/visual-qa) by Avi Singh

* [Open Domain Question Answering: Techniques, Resources and Systems](http://lml.bas.bg/ranlp2005/tutorials/magnini.ppt) by Bernardo Magnini

* [Question Answering Techniques for the World Wide Web](https://cs.uwaterloo.ca/~jimmylin/publications/Lin_Katz_EACL2003_tutorial.pdf) by Lin Katz, University of Waterloo, Canada

* [NLP-Question-Answer-System](https://github.com/raoariel/NLP-Question-Answer-System/blob/master/simpleQueryAnswering.py)

	> Built from scratch using corenlp and nltk for sentence segmenting and POS tagging

* [PiQASso: Pisa Question Answering System](http://trec.nist.gov/pubs/trec10/papers/piqasso.pdf) by Attardi et al., 2001

	> Uses traditional information retrieval (IR) NLP

### Deep learning

* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs) by Christopher Olah

	> A clear and correct explanation of LSTMs

* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf) by Kyunghyun Cho et al., 2014

	> Paper that first introduced gated recurrent units, making LSTMs more efficient for NLP

### LSTMs and RNNs

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf) by Cho et al.

	> Explains how the contents of the memory cells in an LSTM layer can be used as an embedding that can encode variable length sequences and then decode them to a new variable length sequence with a potentially different length, translating or transcoding one sequence into another.

* [Reinforcement Learning with Long Short-Term Memory](https://papers.nips.cc/paper/1953-reinforcement-learning-with-long-short-term-memory.pdf) by Bram Bakker

	> Application of LSTMs to planning and anticipation cognition with demonstrations of a network that can solve the T-maze navigation problem and an advanced pole-balancing (inverted pendulum) problem.

* [Supervised Sequence Labelling with Recurrent Neural Networks](https://mediatum.ub.tum.de/doc/673554/file.pdf) Thesis by Alex Graves with advisor B. Brugge
	> a detailed explanation of the mathematics for the exact gradient for LSTMs as first proposed by Hochreiter and Schmidhuber in 1997. But Graves fails to define terms like CEC or LSTM block/cell rigorously.

* [Theano LSTM documentation](http://deeplearning.net/tutorial/lstm.html) by Pierre Luc Carrier and Kyunghyun Cho

	> Diagram and discussion to explain the LSTM implementation in Theano and Keras.

* [Learning to Forget: Continual Prediction with LSTM] (http://mng.bz/4v5V) by Felix A. Gers, Jurgen Schmidhuber, and Fred Cummins

	> Uses nonstandard notation for layer inputs (yin) and outputs (yout) and internal hidden state (h). All math and diagrams are “vectorized.”

* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) by Ilya Sutskever, Oriol Vinyals, and Quoc V. Le at Google.

* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs) 2015 blog by Charles Olah—lots of good diagrams and discussion/feedback from readers.

* [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf) by Sepp Hochreiter and Jurgen Schmidhuber, 1997

	> Original paper on LSTMs with outdated terminology and inefficient implementation, but detailed mathematical derivation.

## 6. Competitions and awards

* [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html)

	> Some researchers believe that compression of natural language text is equivalent to artificial general intelligence (AGI).

* [Hutter Prize](https://en.wikipedia.org/wiki/Hutter_Prize)

	> Annual competition to compress a 100 MB archive of Wikipedia natural language text. Alexander Rhatushnyak won in 2017

* [Open Knowledge Extraction Challenge 2017](https://svn.aksw.org/papers/2017/ESWC_Challenge_OKE/public.pdf)

## 7. Datasets 

* [https://livebook.manning.com/book/natural-language-processing-in-action/resources/1](https://livebook.manning.com/book/natural-language-processing-in-action/resources/1)

## 8. Search Engine

### Search algorithms

* [Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734.pdf) 

	> BidMACH is a high-dimensional vector indexing and KNN search implementation, similar to the annoy Python package. This paper explains an enhancement for GPUs that is 8 times faster than the original implementation.

* [Spotify’s Annoy Package](https://erikbern.com/2017/11/26/annoy-1.10-released-with-hamming-distance-and-windows-support.html) by Erik Bernhardsson’s

	> A K-nearest neighbors algorithm used at Spotify to find similar songs.

* [New benchmarks for approximate nearest neighbors by Erik Bernhardsson](https://erikbern.com/2018/02/15/new-benchmarks-for-approximate-nearest-neighbors.html)

	> Approximate nearest neighbor algorithms are the key to scalable semantic search, and author Erik keeps tabs on the state of the art.

### Open source search engines

* [BeeSeek](https://launchpad.net/~beeseek-devs)

	> Open source distributed web indexing and private search (hive search); no longer maintained

* [WebSPHNIX](https://www.cs.cmu.edu/~rcm/websphinx/)

	> Web GUI for building a web crawler

### Open source full-text indexers

* [Elasticsearch](https://github.com/elastic/elasticsearch)

	> Open Source, Distributed, RESTful Search Engine.

* [Apache Lucern + Solr](https://github.com/apache/lucene-solr)
* [Sphinx Search](https://github.com/sphinxsearch/sphinx)
* [Kronuz/Xapiand: Xapiand: A RESTful Search Engine](https://github.com/Kronuz/Xapiand)

	> There are packages for Ubuntu that’ll let you search your local hard drive (like Google Desktop used to do)

* [Indri](http://www.lemurproject.org/indri.php)

* [Semantic search with a Pyt-hon interface](https://github.com/cvangysel/pyndri), but it isn’t actively maintained.

* [Gigablast](https://github.com/gigablast/open-source-search-engine)
	
	> Open source web crawler and natural language indexer in C++.
	
* [Zettair](http://www.seg.rmit.edu.au/zettair)

	> Open source HTML and TREC indexer (no crawler or live example); last updated 2009

* [OpenFTS: Full Text Search Engine](http://openfts.sourceforge.net)

* [Full text search indexer for PyFTS using PostgreSQL with a Python API](http://rhodesmill.org/brandon/projects/pyfts.html)

### Manipulative search engines

> The search engines most of us use aren’t optimized solely to help you find what you need, but rather to ensure that you click links that generate revenue for the company that built it. Google’s innovative second-price sealed-bid auction ensures that advertisers don’t overpay for their ads,[1] but it doesn’t prevent search users from overpaying when they click disguised advertisements. This manipulative search isn’t unique to Google. It’s used in any search engine that ranks results according to any “objective function” other than your satisfaction with the search results. But here they are, if you want to compare and experiment

* [Cornell University Networks Course case study, “Google AdWords Auction - A Second Price Sealed-Bid Auction”](https://blogs.cornell.edu/info2040/2012/10/27/google-adwords-auction-a-second-price-sealed-bid-auction)

### Less manipulative search engines

> To determine how commercial and manipulative a search engine was, I queried several engines with things like “open source search engine.” I then counted the number of ad-words purchasers and click-bait sites among the search results in the top 10. The following sites kept that count below one or two. And the top search results were often the most objective and useful sites, such as Wikipedia, Stack Exchange, or reputable news articles and blogs

* [Alternatives to Google](https://www.lifehack.org/374487/try-these-15-search-engines-instead-google-for-better-search-results).

	> See the web page titled “[Try These 15 Search Engines Instead of Google For Better Search Results](https://www.lifehack.org/374487/try-these-15-search-engines-instead-google-for-better-search-results)”.

* [Yandex](https://yandex.com/search/?text=open%20source%20search%20engine&lr=21754)

	> Surprisingly, the most popular Russian search engine (60% of Russian searches) seemed less manipulative than the top US search engines

* [DuckDuckGo](https://duckduckgo.com)
* [Watson Semantic Web Search](http://watson.kmi.open.ac.uk/WatsonWUI)

	> No longer in development, and not really a full-text web search, but it’s an interesting way to explore the semantic web (at least what it was years ago before watson was frozen).

### Distributed search engines

> Distributed search engines[3] are perhaps the least manipulative and most “objective” because they have no central server to influence the ranking of the search results. However, current distributed search implementations rely on TF-IDF word frequencies to rank pages, because of the difficulty in scaling and distributing semantic search NLP algorithms. However, distribution of semantic indexing approaches such as latent semantic analysis (LSA) and locality sensitive hashing have been successfully distributed with nearly linear scaling (as good as you can get). It’s just a matter of time before someone decides to contribute code for semantic search into an open source project like Yacy or builds a new distributed search engine capable of LSA

* [Nutch](https://nutch.apache.org/)

	> Nutch spawned Hadoop and itself became less of a distributed search engine and more of a distributed HPC system over time

* [Yacy](https://www.yacy.net/en/index.html)

	> One of the few open source [`https://github.com/yacy/yacy_search_server`](https://github.com/yacy/yacy_search_server) decentralized, or federated, search engines and web crawlers still actively in use. Preconfigured clients for Mac, Linux, and Windows are available.




