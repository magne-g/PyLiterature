import pandas as pd
import numpy as np
import sklearn as s
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.externals.six import StringIO
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from datetime import date
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Settings

_print_unprocessed_dataset_stats = False
_print_processed_dataset_stats = False
_verbose = 1


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_info_columns', 500)

dot_data = StringIO()
rng = np.random.RandomState(2)



def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def result(scores, model, param_score, label, predictions):
    print("Accuracy: %0.4f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    ax = sns.regplot(Y_test, predictions, fit_reg=True)
    ax.set(xlabel='True price in NOK/1000', ylabel='Predicted price in NOK/1000')


def pandas_encoding(data):
    data['manufacturer'] = data['manufacturer'].astype('category').cat.codes
    data['model'] = data['model'].astype('category').cat.codes
    data['gear'] = data['gear'].astype('category').cat.codes
    data['trans'] = data['trans'].astype('category').cat.codes
    data['fuel_type'] = data['fuel_type'].astype('category').cat.codes
    return data

def print_dataset_stats(data):
    print(data.describe())
    print(data.count())
    print(data.corr())

def label_encode(data):
    le = preprocessing.LabelEncoder()
    data['manufacturer'] = le.fit_transform(data['manufacturer'].str.lower())
    data['model'] = le.fit_transform(data['model'].str.lower())
    data['gear'] = le.fit_transform(data['gear'].str.lower())
    data['trans'] = le.fit_transform(data['trans'].str.lower())
    data['fuel_type'] = le.fit_transform(data['fuel_type'].str.lower())

    return data


#TODO Move preprocessing
def preprocess(data):

    if _print_unprocessed_dataset_stats:
        print('Unprocessed Dataset Statistics:')
        print_dataset_stats(data)


    data = data.set_index('finn_code')
    data['price'] = data['price'].astype(np.float)
    data['price'] = np.round(data['price'], -3)
    data['price'] = np.divide(data['price'], 1000)
    data['km'] = data['km'].astype(np.float)
    data['km'] = np.round(data['km'], -3)
    data['km'] = np.divide(data['km'], 1000)
    data = data[data.model_year > 1985]
    data = data[data.price < 950]
    data = data[data.price > 15]
    data = data[(((data.model_year >= 2017) & (data.price > 15)) | (data.model_year < 2017))]
    data.loc[data.cylinder > 10, 'cylinder'] = np.round(data.cylinder/1000)
    data.loc[data.fuel_type == 'Elektrisitet', 'cylinder'] = 0
    data = data[data.km < 290]
    data = data[data.power > 0]
    data = data[data.power < 500]

    data['model_year'] = date.today().year - data['model_year']

    indices = data[(data['fuel_type'] == 'Diesel') & (data['cylinder'] == 0)].index
    data.drop(indices, inplace=True)
    indices = data[(data['fuel_type'] == 'Bensin') & (data['cylinder'] == 0)].index
    data.drop(indices, inplace=True)
    data = data.drop(columns=['first_reg'])
    # Dropping rows with null in one or more attribute:
    #data = data.sample(3000)
    #data['cylinder'] = data['cylinder'].fillna(data.cylinder.mean())
    data = data.dropna()
    #data = data.sort_values(by=['km'], axis=0)
    label_encode(data)

    if _print_processed_dataset_stats:
        print('Processed Dataset Statistics:')
        print_dataset_stats(data)

    return data

def tune_parameter(param, value, data):
    tuned_data = data[data[param] < value]
    print('Evaluating parameter "' + param + ' with value: ', value)

    return tuned_data

best_param_score = [0.0,0.0]

def process_and_label(data, dependant_variable):

    data.index = data.index.astype(int)  # use astype to convert to int
    pre_count = len(data)
    data = preprocess(data)
    label = data[dependant_variable.name]
    data = data.drop(columns=label.name)
    data.to_csv('../labeled_cars.csv')
    post_count = len(data)
    if(_verbose > 0):
        print('{0:.0f}% of data rows dropped in processing...'.format(100 - (post_count/pre_count * 100)))
    return data, label

def split_data(data, label, test_size=0.2):

    X_train, X_test, Y_train, Y_test = train_test_split(data,
                                                        label,
                                                        test_size=test_size,
                                                        random_state=rng)
    return X_train, X_test, Y_train, Y_test

def define_estimator():
    estimator = AdaBoostRegressor(RandomForestRegressor(random_state=rng,
                                                        n_estimators=30,
                                                        n_jobs=8),
                                  random_state=rng,
                                  n_estimators=5)

    return estimator


def fit_model(X_train, Y_train, estimator):

    model = estimator.fit(X_train, Y_train)

    return model

def load_dataset(filename):
    data = pd.read_csv(filename)
    if(_verbose > 0):
        print('Loaded {count} rows of data from {filename}'.format(count=len(data), filename=filename))
    data_orig = data
    return data, data_orig




data, data_orig = load_dataset('../processed_cars.csv')

data, label = process_and_label(data, dependant_variable=data['price'])

if(len(data) != len(label)):
    print(len(data), len(label))
    assert(len(data) == len(label), 'Error: Length not equal in X, Y')

X_train, X_test, Y_train, Y_test = split_data(data, label)

estimator = define_estimator()

model = fit_model(X_train, Y_train, estimator=estimator)

scores = cross_val_score(model, X_test, Y_test, cv=3)

predictions = model.predict(X_test)

result(scores, model, best_param_score, label, predictions)

plt.show()





