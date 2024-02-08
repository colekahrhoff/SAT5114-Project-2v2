import sklearn
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 
import numpy as np 

#Converting dataset into workable dataframe and seperating into features and target for processing. Primarily using the dataset itself. 
cancer = datasets.load_breast_cancer()
data = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
data['Cancer'] = pd.DataFrame(cancer['target'])
info = data.drop('Cancer', axis = 1).values
target = data['Cancer'].values

#Split the data
x_train, x_test, y_train, y_test = train_test_split(info, target, test_size = 0.1)

#Use grid search to find the most accurate
folds = KFold(n_splits=10)
param_grid = {'n_neighbors':(range(1,51))}
clas = KNeighborsClassifier()
grid = GridSearchCV(clas, param_grid, cv=folds, scoring = 'accuracy')
grid.fit(x_train, y_train)
param = grid.best_params_['n_neighbors']

#Finally, perform the model using the best k neighbors parameter
model = KNeighborsClassifier(n_neighbors = param)
scores = cross_val_score(model, x_test, y_test, cv=folds, scoring = 'accuracy')
print(scores)