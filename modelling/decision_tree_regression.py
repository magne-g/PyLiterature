import pandas as pd
import numpy as np
import sklearn as s
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.externals.six import StringIO
import seaborn as sns
import pydot
from sklearn import metrics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_info_columns', 500)

dot_data = StringIO()
rng = np.random.RandomState(1)

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def pandas_encoding(data):
    data['manufacturer'] = data['manufacturer'].astype('category').cat.codes
    data['model'] = data['model'].astype('category').cat.codes
    data['gear'] = data['gear'].astype('category').cat.codes
    data['trans'] = data['trans'].astype('category').cat.codes
    data['fuel_type'] = data['fuel_type'].astype('category').cat.codes
    return data

def label_encoder(data):
    le = preprocessing.LabelEncoder()
    data['manufacturer'] = le.fit_transform(data['manufacturer'].str.lower())
    data['model'] = le.fit_transform(data['model'].str.lower())
    data['gear'] = le.fit_transform(data['gear'].str.lower())
    data['trans'] = le.fit_transform(data['trans'].str.lower())
    data['fuel_type'] = le.fit_transform(data['fuel_type'].str.lower())

    return data

def preprocess(data):
    data = data.dropna()
    data = data.set_index('finn_code')

    data['price'] = data['price'].astype(np.float)
    data['price'] = np.round(data['price'], -3)
    data['price'] = np.divide(data['price'], 1000)

    data = data[data.price < 750]
    data = data[data.price > 15]
    data = data[data.cylinder > 0]
    data = data[data.cylinder < 10]
    data = data[data.km < 300000]
    data = data[data.power < 500]
    data = data[data.power > 0]

    # data = data.sample(2500)
    # data = pandas_encoding(data)
    label_encoder(data)

    #data = data.drop(columns=['finn_code'])
    #data = data.drop(columns=['fuel_type'])
    #data = data.drop(columns=['gear'])
    #data = data.drop(columns=['manufacturer'])
    #data = data.drop(columns=['trans'])
    #data = data.drop(columns=['cylinder'])
    #data = data.drop(columns=['first_reg'])
    #data = data.drop(columns=['model'])
    #data = data.drop(columns=['km'])

    print(data.describe())
    print(data.count())
    print(data.corr())

    return data


data = pd.read_csv('../processed_cars.csv')
validation_data  = pd.read_csv('../processed_cars_test.csv')


data = preprocess(data)
validation_data = preprocess(validation_data)
#Dropping rows with null in one or more attribute:



label = data['price']

data = data.drop(columns=['price'])
validation_data = validation_data.drop(columns=['price'])


X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#tree = AdaBoostRegressor(DecisionTreeRegressor(criterion='mse'), n_estimators=150, random_state=rng)

tree = GridSearchCV(DecisionTreeRegressor(max_depth=10, max_features=6, min_weight_fraction_leaf=0.001, max_leaf_nodes=326), param_grid={'presort': [True, False]}, cv=7)

model = tree.fit(X_train, Y_train)


scores = cross_val_score(model, data, label, cv=3)

predictions = tree.predict(X_test)

#scores = cross_validate(model, data, label, cv=15)

#print("Feature importance: ", model.feature_importances_)

print("Params: ", model.get_params())
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
print(validation_data)
print(model.predict(validation_data))

#s.tree.export_graphviz(model, out_file=dot_data)

ax = sns.regplot(Y_test, predictions, fit_reg=True)

ax.set(xlabel='True price in NOK/1000', ylabel='Predicted price in NOK/1000')

#model.decision_path(validation_data.sample(1))
print(model.cv_results_)

print("Best Estimator:", model.best_estimator_)
print("Best Index: ", model.best_index_)
print("Best Params: ", model.best_params_)
print("Best Score: ", model.best_score_)


plt.show()
