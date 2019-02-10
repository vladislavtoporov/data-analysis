import graphviz
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import misc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import decomposition
from sklearn import datasets

# load dataset
data = pd.read_csv('data/data.csv').drop(columns=['Unnamed: 0', 'artist', 'song_title'])
data.describe()
features = ["valence", "energy", "danceability", "speechiness", "acousticness", "instrumentalness", "loudness","duration_ms","liveness","tempo","time_signature","mode","key"]
X = data.drop(columns={'target'})
y = data.target

# run PCA
pca = decomposition.PCA(n_components=7)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered, y)
X_pca = pca.transform(X_centered)


#split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, shuffle=True)

# Build a simple Decision Tree Classifier based on a set of features¶
clf = DecisionTreeClassifier(min_samples_leaf=26, random_state=15)
clf.fit(X_train, y_train)
# Run prediction on test data
y_pred = clf.predict(X_test)

# calc Accuracy
score = accuracy_score(y_test, y_pred) * 100
rounded_score = round(score, 1)
print("Decision Tree Classifier Accuracy: {}%".format(rounded_score))
