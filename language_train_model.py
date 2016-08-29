#Language Detector Model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron 
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pickle

#load training data
txt_data_folder = '/Users/stephanieo/Desktop/MLProjects/European-language-classification/txt_short/'
dataset = load_files(txt_data_folder, shuffle = True)

#split data into test
docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.5)

#Vectorizer to split text into n-grams for n \in {1, 2, 3}

tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, 3))

#Pipeline using vectorizer and Perceptron?
clf = Pipeline([('vect', tfidf_vectorizer),
				('lin', Perceptron()),
	])

#train data
_ = clf.fit(docs_train, y_train)

#predict test data
y_predicted = clf.predict(docs_test)

#pickle classifier
# s = pickle.dumps(clf)

#Output metrics & plot the confusion matrix 
print(metrics.classification_report(y_test, y_predicted,
                                    target_names=dataset.target_names))

cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)





