from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)
# load dataset
df = pd.read_csv("data/data.csv").drop(columns=['Unnamed: 0', 'artist', 'song_title'])
X = df.drop(columns={'target'})
y = df.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

# create pipeline
sc = StandardScaler()
pca = PCA()
log = LogisticRegression()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
pipe = Pipeline(steps=[('sc', sc),
                       ('pca', pca),
                       # ('logistic', log),
                       ('tree', tree),
                       # ('forest', forest),
                       ])

# create default parameners
n_components = list(range(1, X_train.shape[1] + 1, 1))
C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']
tree__random_state = list(range(0, 60, 5))
tree__min_samples_leaf = list(range(1, 60, 5))
forest__n_estimators = n_estimators = list(range(10, 90, 10))

parameters = dict(pca__n_components=n_components,
                  # logistic__C=C,
                  # logistic__penalty=penalty,
                  tree__random_state=tree__random_state,
                  tree__min_samples_leaf=tree__min_samples_leaf,
                  # forest__n_estimators=forest__n_estimators,
                  # forest__random_state=tree__random_state,
                  # forest__min_samples_leaf=tree__min_samples_leaf,
                  )

# Create Grid Search
clf = GridSearchCV(pipe, parameters)
# Conduct Grid Search
clf.fit(X_train, y_train)

# calc absolute error
from sklearn.utils import shuffle

new_random_test_X = pd.concat([X_train, X_test])
new_random_test_y = pd.concat([y_train, y_test])
y_pre = clf.predict(new_random_test_X)

mea = mean_absolute_error(new_random_test_y, y_pre)
print("absolute_error", mea)

cross_val_score = cross_val_score(clf, new_random_test_X, new_random_test_y)
print("cross val score", cross_val_score)

# print('Best Penalty:', clf.best_estimator_.get_params()['logistic__penalty'])
# print('Best C:', clf.best_estimator_.get_params()['logistic__C'])

print('Best min_samples_leaf:', clf.best_estimator_.get_params()['tree__min_samples_leaf'])
print('Best random_state:', clf.best_estimator_.get_params()['tree__random_state'])
# print('Best n_estimators:', clf.best_estimator_.get_params()['forest__n_estimators'])
# print('Best min_samples_leaf:', clf.best_estimator_.get_params()['forest__min_samples_leaf'])
# print('Best random_state:', clf.best_estimator_.get_params()['forest__random_state'])

print('Best Number Of Components:', clf.best_estimator_.get_params()['pca__n_components'])
