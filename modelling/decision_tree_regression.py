import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals.six import StringIO
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, learning_curve, train_test_split

InteractiveShell.ast_node_interactivity = "all"
warnings.simplefilter(action='ignore', category=FutureWarning)

# Settings

_print_unprocessed_dataset_stats = False
_print_processed_dataset_stats = False
_tune_hyper_parameters = True
_use_one_hot_encoding = True
_show_plots = True
_exclude_param_tuning = False
_verbose = 1

_important_data = '{}'
alpha = 1.0
lasso = Lasso(alpha=alpha)

power_transform_price = preprocessing.PowerTransformer('yeo-johnson')

# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_info_columns', 500)
pd.set_option('use_inf_as_na', True)

dot_data = StringIO()
r_state = np.random.RandomState(42)
X_train, X_test, Y_train, Y_test = [0] * 4


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

    data['manufacturer'] = data['manufacturer'].astype('category')
    data['model'] = data['model'].astype('category')
    data['gear'] = data['gear'].astype('category')
    data['trans'] = data['trans'].astype('category')
    data['fuel_type'] = data['fuel_type'].astype('category')

    return data


def save_accuracy_log(scores, estimator):
    filename = np.round(scores.mean(), 4)

    data = pd.Series(estimator).to_frame()

    # params = pd.DataFrame.from_items(data)

    data.to_csv('../logs/' + str(filename))


def preprocess(data):
    if not _exclude_param_tuning:
        data = data.drop(columns=['first_reg'])
    if _print_unprocessed_dataset_stats:
        print('Unprocessed Dataset Statistics:')
        print_dataset_stats(data)
    if not _exclude_param_tuning:
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
        data.loc[data.fuel_type == 'Elektrisitet', 'cylinder'] = 0
        data = data[data.km < 350]
        data = data[data.power > 0]
        data = data[data.power < 500]
        power_transform_price.fit(data.loc[:, 'price'].values.reshape(-1, 1))
        data['price'] = power_transform_price.transform(data.loc[:, 'price'].values.reshape(-1, 1))
        data['model_age'] = (date.today().year - data['model_year'])
        indices = data[(data['fuel_type'] == 'Diesel') & (data['cylinder'] == 0)].index
        data.drop(indices, inplace=True)
        indices = data[(data['fuel_type'] == 'Bensin') & (data['cylinder'] == 0)].index
        data.drop(indices, inplace=True)

    # kbins = preprocessing.KBinsDiscretizer(encode='ordinal', n_bins=15)
    # kbins.fit(data.loc[:, 'km'].values.reshape(-1,1))
    # data['km'] = kbins.transform(data.loc[:, 'km'].values.reshape(-1, 1))

    # global power_transform_price

    # power_transform_price = preprocessing.MinMaxScaler(
    #    [np.log(data.price.min()),
    #     np.log(data.price.max())])

    # power_transform_price = preprocessing.MinMaxScaler(
    #    [np.log(data.price.min()),
    #     np.log(data.price.max())])

    # power_transform_price = preprocessing.power_transform('yeo-johnson')

    # power_transform = preprocessing.MinMaxScaler(
    #    [np.log(data.price.min()),
    #     np.log(data.price.max())])

    # power_transform_age = preprocessing.PowerTransformer('yeo-johnson')
    # plot_dist(data['model_age'], title='Distribution Before - model_age')
    # data['model_age'] = power_transform_age.fit_transform(data.loc[:, 'model_age'].values.reshape(-1, 1))

    data = data.dropna()
    # imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=2)
    # imputer.fit(data.loc[:, 'cylinder'].values.reshape(-1, 1))

    # data['cylinder'] = imputer.transform(data.loc[:, 'cylinder'].values.reshape(-1, 1))

    # imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=2)
    # imputer.fit(data.loc[:, 'cylinder'].values.reshape(-1, 1))
    # data['cylinder'] = imputer.transform(data.loc[:, 'cylinder'].values.reshape(-1, 1))
    print(data.info(verbose=True))

    # es = ft.EntitySet(id='cars')

    # es = es.entity_from_dataframe(entity_id='cars', dataframe=data, index='finn_code')

    # es = es.normalize_entity(base_entity_id="cars",
    #                         new_entity_id="model",
    #                         index="manufacturer",
    #                         additional_variables=["power"])

    # es = es.add_interesting_values(verbose=True)

    # new_relationship = ft.Relationship(es['model']['manufacturer'])
    # es = es.add_relationship(new_relationship)

    # data['price'] = label.values

    print(data.info(verbose=True))

    # data = data.sort_values(by=['km'], axis=0)
    label_encode(data)

    if (_use_one_hot_encoding and not _exclude_param_tuning):
        one_hot_cols = ['trans', 'fuel_type', 'gear']  # Nominals with low cardinality
        data = pd.get_dummies(data, columns=one_hot_cols)

    # data = data.drop(columns=['model'])
    data = data.drop(columns=['model_year'])
    # data = data.drop(columns=['manufacturer'])
    # data = data.drop(columns=['fuel_type'])
    # data = data.drop(columns=['gear'])
    # data = data.drop(columns=['cylinder'])

    if _print_processed_dataset_stats:
        print('Processed Dataset Statistics:')
        print_dataset_stats(data)

    return data


def tune_parameter(param, value, data):
    tuned_data = data[data[param] < value]
    print('Evaluating parameter "' + param + ' with value: ', value)

    return tuned_data


best_param_score = [0.0, 0.0]


def process_and_label(data, dependant_variable):
    data.index = data.index.astype(int)  # use astype to convert to int
    # data = data.sample(1500)
    pre_count = len(data)
    data = preprocess(data)
    label = data[dependant_variable.name]
    data = data.drop(columns=label.name)

    print(data)
    data.to_csv('../labeled_cars.csv')
    post_count = len(data)
    if (_verbose > 0):
        print('{0:.0f}% of data rows dropped in processing...'.format(100 - (post_count / pre_count * 100)))
    return data, label


def split_data(data, label, test_size=0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(data,
                                                        label,
                                                        test_size=test_size,
                                                        random_state=r_state)
    return X_train, X_test, Y_train, Y_test


def define_estimator(d, _base_estimator, _meta_estimator, _tune_hyper_parameters):
    _estimator = _base_estimator

    print(_estimator)
    if not _tune_hyper_parameters:
        if (_meta_estimator):
            return _meta_estimator(_estimator)
        else:
            return (_estimator)

    else:
        parameters = [{'n_estimators': [300],
                       'max_depth': [15],
                       'max_features': ['auto', 'sqrt', 'log2'],
                       'min_samples_split': [2],
                       'min_samples_leaf': [1]
                       }]
        print('Starting GridSearchCV with params: ', parameters)

        _cv = GridSearchCV(_base_estimator, param_grid=parameters, n_jobs=8, verbose=2, cv=3, scoring='r2', refit=True)

        _cv.fit(d['X'], d['Y'])

        print(_cv.cv_results_)
        print(_cv.best_params_)
        print(_cv.best_score_)

    _estimator = _estimator.set_params(**_cv.best_params_)

    return _estimator


def plot_learning_curve(estimator, title, X, y, ylim=(0.85, 1), cv=None,
                        n_jobs=None, train_sizes=np.logspace([0.1, 1.0], num=50, stop=1.0, endpoint=True, base=0.01)):
    plt.figure()
    plt.title(title)
    if _exclude_param_tuning:
        ylim = (0.5, 1)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2', random_state=r_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.ylabel("Score " + str(np.amax(test_scores)))
    plt.grid()
    plt.xscale('log')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def fit_model(X_train, Y_train, estimator):
    model = estimator.fit(X_train, Y_train)

    return model


def load_dataset(filename):
    data = pd.read_csv(filename)
    if (_verbose > 0):
        print('Loaded {count} rows of data from {filename}'.format(count=len(data), filename=filename))
    data_orig = data
    return data, data_orig


def show_boxplot():
    g = sns.boxplot('manufacturer', 'price', data=data)
    locs, labels = plt.xticks()
    fig = plt.gcf()
    fig.set_size_inches(16, 9)

    g.axes.set_title("Title", fontsize=50)
    g.tick_params(labelsize=12)
    plt.setp(labels, rotation=90)
    plt.show()


def plot_dist_before(title):
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.distplot(data.price)
    ax.set_title(title)
    plt.show()


def plot_dist_after(title):
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.distplot(label)
    ax.set_title(title)
    plt.show()


def plot_dist(x, title):
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.distplot(x)
    ax.set_title(title)
    plt.show()


def show_table_pre_proccesing():
    demo_table = data.head()


cwd = os.getcwd()
sub = cwd.split('/')
sub = sub[:-1]
path = '/'.join(sub)
path += '/'
path += 'processed_cars.csv'
filename = path

data, data_orig = load_dataset(filename)

show_boxplot()

plot_dist_before('Price Distribution - Before Transformation')

data, label = process_and_label(data, dependant_variable=data['price'])

print(data.info(verbose=True))

plot_dist_after('Price Distribution - After Transformation')

if (len(data) != len(label)):
    print(len(data), len(label))
    assert len(data) == len(label), 'Error: Length not equal in X, Y'

X, x, Y, y = split_data(data, label)

d = {'X': X, 'x': x, 'Y': Y, 'y': y}

# estimator = define_estimator(d, RandomForestRegressor(random_state=rng, n_jobs=8), AdaBoostRegressor(n_estimators=5),_tune_hyper_parameters=True)


estimator = GradientBoostingRegressor(loss='ls',
                                      alpha=0.9,
                                      max_depth=5,
                                      max_features='auto',
                                      n_estimators=300,
                                      random_state=r_state,
                                      learning_rate=0.1,
                                      min_samples_split=2,
                                      min_samples_leaf=1)

print(estimator.get_params())

alpha_range = frange(0.1, 1.5, 0.1)

param_grid = {'loss': ['huber', 'ls', 'lad', 'quantile'],
              'alpha': np.arange(0.05, 0.95, 0.01),
              'learning_rate': np.arange(0.05, 0.95, 0.01),
              'min_samples_split': np.arange(2, 8, 1),
              'min_samples_leaf': np.arange(1, 5, 1),
              'max_depth': np.arange(3, 9, 1)}

# rcv = RandomizedSearchCV(estimator, param_grid, n_jobs=8, random_state=rng, verbose=1,
# cv = KFold(n_splits=10, random_state=rng))

# rcv.fit(X, Y)

# print(rcv.best_params_)
# print(rcv.best_score_)
# print(rcv.best_index_)
# print(rcv.best_estimator_)

# estimator = rcv.best_estimator_


plot_learning_curve(estimator, 'Learning Curves - GradientBoostingRegressor', X, Y, (.75, 1.01),
                    cv=KFold(n_splits=10, random_state=r_state), n_jobs=8)

plt.show()


model = fit_model(X, Y, estimator=estimator)

scores = cross_val_score(model, d['x'], d['y'], cv=KFold(n_splits=10, random_state=r_state), n_jobs=8, scoring='r2')

predictions = model.predict(d['x'])

scores_training_data = cross_val_score(model, d['X'], d['Y'], cv=KFold(n_splits=10, random_state=r_state), n_jobs=8,
                                       scoring='r2')

if not _exclude_param_tuning:
    predicted_prices = power_transform_price.inverse_transform(np.array(predictions).reshape(-1, 1))
    seen_prices = power_transform_price.inverse_transform(np.array(y).reshape(-1, 1))
    seen_prices = np.array(seen_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()
else:
    predicted_prices = predictions
    seen_prices = y

_accuracy = "Accuracy Mean: %0.4f (+/- %0.3f)" % (
    power_transform_price.inverse_transform(
        np.mean(np.array(
            scores
        )
    ).reshape(-1, 1)), scores.std() * 2)

_accuracy_max = "Accuracy Max: %0.4f" % (scores.max())

_accuracy_training = "Training Accuracy: %0.4f (+/- %0.3f)" % (
scores_training_data.mean(), scores_training_data.std() * 2)

ax = sns.regplot(seen_prices, predicted_prices, fit_reg=True, robust=True)
ax.set(xlabel='True price', ylabel='Predicted price')
ax.set_title(_accuracy)

print(_accuracy)
print(_accuracy_max)
print(_accuracy_training)

save_accuracy_log(scores, estimator)

plt.show()

feat_importances = pd.Series(model.feature_importances_, index=data.columns)
feat_importances.nlargest(6).plot(kind='barh')

plt.show()

print(model.get_params())
print(x.columns)
print(model.feature_importances_)

print(data.info(verbose=True))

print(data.corr())

full_data = data
full_data['price'] = power_transform_price.inverse_transform(np.array(label).reshape(-1, 1))

sample = full_data.sample(999)
sample.to_csv('../sample.csv')

print(predicted_prices)
print(d['x'])

# run(path, X_train, Y_train, X_test, Y_test)
