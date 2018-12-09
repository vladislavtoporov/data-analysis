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

np.random.seed(0)
# load dataset
df = pd.read_csv("data/data.csv").drop(columns={'Unnamed: 0'})
X = df.drop(columns={'target', 'song_title', 'artist'})
y = df.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

# create pipeline
sc = StandardScaler()
pca = PCA()
log = LogisticRegression()
lin = LinearRegression()

pipe = Pipeline(steps=[('sc', sc),
                       ('pca', pca),
                       ('logistic', log),
                       # ('linear', lin),
                       ])

# create default parameners
n_components = list(range(1,X_train.shape[1]+1,1))
print(n_components)
C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']
parameters = dict(pca__n_components=n_components,
                  logistic__C=C,
                  logistic__penalty=penalty)

# Create Grid Search
clf = GridSearchCV(pipe, parameters)
# Conduct Grid Search
clf.fit(X_train, y_train)

# calc absolute error
from sklearn.utils import shuffle
new_df = shuffle(df)
nX_test = new_df.drop(columns=['target','song_title','artist'])
nY_test = new_df.target
new_random_test_X = pd.concat([nX_test, X_test])
new_random_test_y = pd.concat([nY_test, y_test])
y_pre = clf.predict(new_random_test_X)

mea = mean_absolute_error(new_random_test_y, y_pre)
print("absolute_error", mea)

cross_val_score(clf, new_random_test_X, new_random_test_y)
print("cross val score", cross_val_score)


print('Best Penalty:', clf.best_estimator_.get_params()['logistic__penalty'])
print('Best C:', clf.best_estimator_.get_params()['logistic__C'])
print('Best Number Of Components:', clf.best_estimator_.get_params()['pca__n_components'])