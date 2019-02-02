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
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def result(scores, model, i, param_score, label, predictions):
    #print("Params: ", model.get_params())
    print('Tuned param: ', i)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    if(scores.mean() > best_param_score[0]):
        best_param_score[0] = scores.mean()
        best_param_score[1] = i
        print(best_param_score)
    print(validation_data)
    print(model.predict(validation_data))


    # s.tree.export_graphviz(model, out_file=dot_data)

    ax = sns.regplot(label, predictions, fit_reg=True)

    ax.set(xlabel='True price in NOK/1000', ylabel='Predicted price in NOK/1000')


def pandas_encoding(data):
    data['manufacturer'] = data['manufacturer'].astype('category').cat.codes
    data['model'] = data['model'].astype('category').cat.codes
    data['gear'] = data['gear'].astype('category').cat.codes
    data['trans'] = data['trans'].astype('category').cat.codes
    data['fuel_type'] = data['fuel_type'].astype('category').cat.codes
    return data


def label_encode(data):
    le = preprocessing.LabelEncoder()
    data['manufacturer'] = le.fit_transform(data['manufacturer'].str.lower())
    data['model'] = le.fit_transform(data['model'].str.lower())
    data['gear'] = le.fit_transform(data['gear'].str.lower())
    data['trans'] = le.fit_transform(data['trans'].str.lower())
    data['fuel_type'] = le.fit_transform(data['fuel_type'].str.lower())

    return data


def preprocess(data):
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
    data.loc[data.cylinder > 10, 'cylinder'] = np.round(data.cylinder / 1000)
    data.loc[data.fuel_type == 'Elektrisitet', 'cylinder'] = 3
    data = data[data.km < 290]
    data = data[data.power > 0]
    data = data[data.power < 500]

    indices = data[(data['fuel_type'] == 'Diesel') & (data['cylinder'] == 0)].index
    data.drop(indices, inplace=True)
    indices = data[(data['fuel_type'] == 'Bensin') & (data['cylinder'] == 0)].index
    data.drop(indices, inplace=True)
    data = data.drop(columns=['first_reg'])
    # Dropping rows with null in one or more attribute:
    data = data.dropna()
    #data = data.sort_values(by=['km'], axis=0)

    label_encode(data)



    #print(data.describe())
    #print(data.count())
    #print(data.corr())

    return data

def tune_parameter(param, value, data):
    tuned_data = data[data[param] < value]
    print('Evaluating parameter "' + param + ' with value: ', value)
    return tuned_data



best_param_score = [0.0,0.0]

for i in range(1):
    data = pd.read_csv('../processed_cars.csv')
    validation_data = pd.read_csv('../processed_cars_test.csv')

    data_orig = data

    data = preprocess(data)
    validation_data = preprocess(validation_data)
    #data = tune_parameter('power', i, data)

    label = data['price']

    data = data.drop(columns=['price'])
    validation_data = validation_data.drop(columns=['price'])

    # data = data.drop(columns=['finn_code'])
    # data = data.drop(columns=['gear'])
    # data = data.drop(columns=['manufacturer'])
    # data = data.drop(columns=['trans'])
    # data = data.drop(columns=['cylinder'])
    # data = data.drop(columns=['first_reg'])
    # data = data.drop(columns=['model'])
    # data = data.drop(columns=['km'])



    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2)



    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components = None)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    """
    """
    pca = PCA(n_components=None)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    """

   # X_train = data.fit_transform(X_train)
   # X_test = data.transform(X_test)

    tree = AdaBoostRegressor(RandomForestRegressor(random_state=rng, n_estimators=30, n_jobs=8), random_state=rng,n_estimators=5)

    print(X_train)


    model = tree.fit(X_train, Y_train)

    scores = cross_val_score(model, data, label, cv=3)

    predictions = cross_val_predict(model, data, label)

    #predictions = tree.predict(X_test)

    # scores = cross_validate(model, data, label, cv=15)

    # print("Feature importance: ", model.feature_importances_)


    # model.decision_path(validation_data.sample(1))
    # print(model.cv_results_)

    # print("Best Estimator:", model.best_estimator_)
    # print("Best Index: ", model.best_index_)
    # print("Best Params: ", model.best_params_)
    # print("Best Score: ", model.best_score_)



    result(scores, model, i, best_param_score, label, predictions)

    plt.show()


#print("Highest accuracy: (%0.3f) on parameter value: %0d" % (best_param_score[0], best_param_score[1]))



