# European language classification
Challenge for startup.ml.

Classify 21 different European languages using the data given by the <a href = "http://www.statmt.org/europarl/">European Parliament Proceedings Parallel Corpus from 1996-2011</a>. 

Scikit-learn is the main tool used here. The data is analyzed using n-grams, in particular, unigrams, bigrams and trigrams. We use a simple tfidf vectorizer combined with perceptron to create a classifier. Only the text from the month of January, over many years, is used in training and testing the data. The F-score was around 0.94 which was surprising. 

Moving forward, I'd like to continue working on the project, optimizing the classifier with better preprocessing, more data and different algorithms. 
