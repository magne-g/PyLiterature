import os
import warnings
from datetime import date

import numpy as np
import sklearn.utils as utils
import sklearn.utils.testing as test
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from keras.layers.core import Dropout
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics as metrics, preprocessing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, \
    ExtraTreesRegressor
from sklearn.externals.six import StringIO
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, LogisticRegression, SGDRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, learning_curve, train_test_split, \
    validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor


InteractiveShell.ast_node_interactivity = "all"
warnings.simplefilter(action='ignore', category=FutureWarning)

# Settings

_print_unprocessed_dataset_stats = False
_print_processed_dataset_stats = False
_tune_hyper_parameters = True
_use_one_hot_encoding = True
_show_plots = True
_exclude_param_tuning = False
_exclude_price_trans = False
_exclude_price_tuning = False
_keras = True
_verbose = 1
regressor = None
_alg = 'random_forest'
# _alg = 'grad_boost_tree'
# _alg = 'support_vector'
# _alg = 'bayes'
# _alg = 'trees'
# _alg = 'neural'
# _alg = 'k_neighbor'

_important_data = '{}'
alpha = 1.0
lasso = Lasso(alpha=alpha)

power_transform_price = preprocessing.PowerTransformer('box-cox')
scale_price = preprocessing.StandardScaler()

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_info_columns', 10)
pd.set_option('use_inf_as_na', True)

dot_data = StringIO()
rng = np.random.RandomState(42)
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


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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
        data.set_index('finn_code')
        data['order'] = np.arange(len(data))

        data['km'] = data['km'].astype(np.float)
        data = data[data.price < 950000]
        data['km'] = np.round(data['km'], -3)
        data['km'] = np.divide(data['km'], 1000)
        data['price'] = np.round(data['price'], -3)
        data['price'] = np.divide(data['price'], 1000)
        data = data[data.model_year > 1985]

        data = data[(((data.model_year >= 2017) & (data.price > 15)) | (data.model_year < 2017))]
        data.loc[data.cylinder > 10, 'cylinder'] = np.round(data.cylinder / 1000)
        data.loc[data.fuel_type == 'Elektrisitet', 'cylinder'] = 0
        data = data[data.km < 350]
        data = data[data.power > 0]
        data = data[data.power < 500]
        # if(not _exclude_price_trans):
        # power_transform_price.fit(data.loc[:, 'price'].values.reshape(-1, 1))
        # power_transform_price._scaler.with_std = True
        # data['price'] = power_transform_price.transform(data.loc[:, 'price'].values.reshape(-1, 1))
        data['model_age'] = (date.today().year - data['model_year'])
        indices = data[(data['fuel_type'] == 'Diesel') & (data['cylinder'] == 0)].index
        data.drop(indices, inplace=True)
        indices = data[(data['fuel_type'] == 'Bensin') & (data['cylinder'] == 0)].index
        data.drop(indices, inplace=True)

    kbins = preprocessing.KBinsDiscretizer(encode='ordinal', n_bins=25)
    kbins.fit(data.loc[:, 'km'].values.reshape(-1,1))
    data['km'] = kbins.transform(data.loc[:, 'km'].values.reshape(-1, 1))

    # global power_transform_price

    min_max = preprocessing.StandardScaler()
   # mileage_interval = preprocessing.KBinsDiscretizer(n_bins=8, encode='onehot')
   # data[['km']] = mileage_interval.fit_transform(data[['km']])
          #data[['km', 'power', 'cylinder']] = min_max.fit_transform(data[['km', 'power', #'cylinder']])

    data[['km', 'power', 'cylinder','order']] = min_max.fit_transform(data[['km', 'power', 'cylinder','order']])




    # power_transform_price = preprocessing.power_transform('yeo-johnson')


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
    #print(data.info(verbose=True))

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

    #print(data.info(verbose=True))

    # data = data.sort_values(by=['km'], axis=0)




    one_hot_cols = ['trans', 'fuel_type', 'gear', 'manufacturer', 'model']  #
    data = pd.get_dummies(data, columns=one_hot_cols)

    #data = data.drop(columns=['model'])
    #data = data.drop(columns=['model_year'])
    #data = data.drop(columns=['manufacturer'])
    #data = data.drop(columns=['fuel_type'])
    #data = data.drop(columns=['gear'])
    #data = data.drop(columns=['trans'])
    data = data.drop(columns=['finn_code'])
    #data = data.drop(columns=['cylinder'])

    pca = PCA(n_components=20)
    print(pca.fit(data))
    print(pca.explained_variance_)
    if _print_processed_dataset_stats:
        print('Processed Dataset Statistics:')
        #print_dataset_stats(data)

    return data


def tune_parameter(param, value, data):
    tuned_data = data[data[param] < value]
    print('Evaluating parameter "' + param + ' with value: ', value)

    return tuned_data


best_param_score = [0.0, 0.0]


def process_and_label(data, dependant_variable):
    data.index = data.index.astype(int)  # use astype to convert to int
    # data = data.sample(1500)

    n = len(data.index)
    #Process all data first
    data = preprocess(data)

    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle Dataset

    n_post = len(data.index)
    label = data[dependant_variable.name]
    #Extract VALIDATION data from SOURCE data
    V_DATA = data.sample(2000, random_state=rng)

    #Extract VALIDATION label from VALIDATION data
    V_LABEL = V_DATA[dependant_variable.name]

    #Drop VALIDATION data from SOURCE data
    data = data.drop(V_DATA.index)

    #Extract TEST data from SOURCE data
    T_DATA = data.sample(2000, random_state=rng)

    #Extract TEST label from TEST data
    T_LABEL = T_DATA[dependant_variable.name]

    #Drop TEST data from SOURCE data
    data = data.drop(T_DATA.index)

    #Assign remainding SOURCE data to TRAINING data
    TR_DATA = data

    #Extract TRAINING label from TRAINING data
    TR_LABEL = TR_DATA[dependant_variable.name]

    n_v = len(V_DATA.index)
    n_t = len(T_DATA.index)
    n_tr = len(TR_DATA.index)

    debug_string = ('Total Rows: %d \nTotal Rows After Processing: %d\nValidation Data Rows: %d\nTest Data Rows: %d\nTraining Data Rows: %d') % (n, n_post, n_v, n_t, n_tr)
    print(data.describe())
    print(debug_string)

    #Remove dependant variables from all datasets
    V_DATA = V_DATA.drop(columns=label.name)
    # Remove dependant variable from TRAINING data
    T_DATA = T_DATA.drop(columns=label.name)
    # Remove dependant variable from TRAINING data
    TR_DATA = TR_DATA.drop(columns=label.name)

    #print(data)
    #data.to_csv('../labeled_cars.csv')

    return V_DATA, V_LABEL, T_DATA, T_LABEL, TR_DATA, TR_LABEL, data


def plot_price_regression(p, s, title):
    ax = sns.regplot(s, p, fit_reg=True)
    ax.set(xlabel='Seen price', ylabel='Predicted price')
    ax.set_title(title)
    plt.show()


def split_data(data, label, test_size=0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(data,
                                                        label,
                                                        test_size=test_size,
                                                        random_state=rng)

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
        estimator, X, y, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2', random_state=rng)
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


def get_estimator_name(estimator):
    return estimator.__class__.__name__


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


def plot_dist_after(title, label):
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


def plot_val_curve(estimator, X, Y):
    param_range = [1, 5, 15, 50]
    param_name = 'n_estimators'
    train_scores, test_scores = validation_curve(
        estimator, X, Y, param_name=param_name, param_range=param_range, scoring='explained_variance')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


cwd = os.getcwd()
sub = cwd.split('/')
sub = sub[:-1]
path = '/'.join(sub)
path += '/'
path += 'processed_cars.csv'
filename = path

data, data_orig = load_dataset(filename)


environment_string = 'Categorical: One-Hot Encoded\n' \
                     'Label: Divided/Rounded\n' \
                     'Continious:Box Cox Transformation' \
                     '\nCardinal:None'

print(environment_string)

#show_boxplot()
#return V_DATA, V_LABEL, T_DATA, T_LABEL, TR_DATA, TR_LABEL




V_DATA, V_LABEL, T_DATA, T_LABEL, TR_DATA, TR_LABEL, data = process_and_label(data, dependant_variable=data['price'])


print(data.info())
print(data.corr())


test.assert_not_in('price', V_DATA.columns), "LABEL DETECTED in Validation DATA!"
test.assert_not_in('price', T_DATA.columns), "LABEL DETECTED in Test DATA!"
test.assert_not_in('price', TR_DATA.columns), "LABEL DETECTED in Training DATA!"



df = pd.DataFrame({'n_rows': [len(V_DATA.values), len(T_DATA.values) , len(TR_DATA.values)],},
                 index=['Validation (20%)' , 'Test (20%)', 'Training (60%)'])
plot = df.plot.pie(y='n_rows', figsize=(6, 6), )

plt.show()


#d = {'X': X, 'x': x, 'Y': Y, 'y': y}

# estimator = define_estimator(d, RandomForestRegressor(random_state=r_state, n_jobs=8), AdaBoostRegressor(n_estimators=5),_tune_hyper_parameters=True)

#if _alg in '_keras':

 #   estimator = GradientBoostingRegressor(loss='ls',
 #                                          alpha=0.9,
 #                                          max_depth=5,
 #                                          max_features='auto',
 #                                          n_estimators=300,
 #                                          random_state=rng,
 #                                          learning_rate=0.1,
 #                                          min_samples_split=2,
 #                                          min_samples_leaf=1)

  #  plot_val_curve()

  #  dummy_estimator = DecisionTreeRegressor()

  #  print(estimator.get_params())

  #  alpha_range = frange(0.1, 1.5, 0.1)

  #  param_grid = {'loss': ['huber', 'ls', 'lad', 'quantile'],
  #                'alpha': np.arange(0.05, 0.95, 0.01),
  #                'learning_rate': np.arange(0.05, 0.95, 0.01),
  #                'min_samples_split': np.arange(2, 8, 1),
  #                'min_samples_leaf': np.arange(1, 5, 1),
  #                'max_depth': np.arange(3, 9, 1)}

    # rcv = RandomizedSearchCV(estimator, param_grid, n_jobs=8, random_state=rng, verbose=1,
    # cv = KFold(n_splits=10, random_state=rng))

    # rcv.fit(X, Y)

    # print(rcv.best_params_)
    # print(rcv.best_score_)
    # print(rcv.best_index_)
    # print(rcv.best_estimator_)

    # estimator = rcv.best_estimator_

  #  plot_learning_curve(estimator, 'Learning Curves - GradientBoostingRegressor', X, Y, (.75, 1.01),
                       # cv=KFold(n_splits=10, random_state=rng), n_jobs=8)

    #plt.show()

    #model = fit_model(X, Y, estimator=estimator)
    # dummy_model = fit_model(X, Y, dummy_estimator)

    #scores_r2 = cross_val_score(model, d['x'], d['y'], cv=KFold(n_splits=10, random_state=rng), #n_jobs=8, scoring='r2')
    #scores_variance = cross_val_score(model, d['x'], d['y'], cv=KFold(n_splits=10, random_state=rng), #n_jobs=8,
                                     # scoring='explained_variance')
    #scores_mse = cross_val_score(model, d['x'], d['y'], cv=KFold(n_splits=10, random_state=rng), n_jobs=8,
                                 #scoring='neg_mean_squared_error')

    # dummy_scores_r2 = cross_val_score(dummy_model, d['x'], d['y'], cv=KFold(n_splits=20, random_state=r_state), n_jobs=8,
    #                                  scoring='r2')
    # dummy_scores_variance = cross_val_score(dummy_model, d['x'], d['y'], cv=KFold(n_splits=20, random_state=r_state),
    #                                        n_jobs=8, scoring='explained_variance')
    # dummy_scores_mse = cross_val_score(dummy_model, d['x'], d['y'], cv=KFold(n_splits=20, random_state=r_state), n_jobs=8,
    #                                   scoring='neg_mean_squared_error')
    # scores_abs = cross_val_score(model, d['x'], d['y'], cv=KFold(n_splits=10, random_state=rng), n_jobs=8,
    #                              scoring='neg_median_absolute_error')
    # # dummy_scores_mse = np.array(dummy_scores_mse)
    # # dummy_scores_r2 = np.array(dummy_scores_r2)
    # # dummy_scores_variance = np.array(dummy_scores_variance)
    #
    # # dummies = pd.DataFrame()
    #
    # # dummies['mse'] = dummy_scores_mse
    # # dummies['r2'] = dummy_scores_r2
    # # dummies['var'] = dummy_scores_variance
    #
    # # sns.lineplot(data=dummies)
    #
    # # plt.show()
    #
    # # sns.lineplot([dummy_scores_mse, dummy_scores_r2, dummy_scores_variance], [np.linspace(0.0, 1.0, 100)])
    # # plt.show()
    #
    # predictions = model.predict(d['x'])
    #
    # scores_training_data = cross_val_score(model, d['X'], d['Y'], cv=KFold(n_splits=10, random_state=rng), n_jobs=8,
    #                                        scoring='r2')
    #
    # if not _exclude_price_trans:
    #     try:
    #         predicted_prices = power_transform_price.inverse_transform(np.array(predictions).reshape(-1, 1))
    #         seen_prices = power_transform_price.inverse_transform(np.array(y).reshape(-1, 1))
    #         seen_prices = np.array(seen_prices).flatten()
    #         predicted_prices = np.array(predicted_prices).flatten()
    #     except:
    #         raise Exception
    # else:
    #     predicted_prices = predictions
    #     seen_prices = y
    #
    # dummy_name = get_estimator_name(dummy_estimator)
    # name = get_estimator_name(model)
    #
    # _accuracy = "%s | R2: %0.4f (+/- %0.3f)" % (name, scores_r2.mean(), scores_r2.std() * 2)
    # # _accuracy_dummy = "%s | R2: %0.4f (+/- %0.3f)" % (dummy_name, dummy_scores_r2.mean(), dummy_scores_r2.std() * 2)
    #
    # _accuracy_variance = "Explained Variance: %0.4f (+/- %0.3f)" % (scores_variance.mean(), scores_variance.std() * 2)
    # _accuracy_mse = "MSE: %0.4f (+/- %0.3f)" % (scores_mse.mean(), scores_mse.std() * 2)
    #
    # _accuracy_max = "Accuracy Max: %0.4f" % (scores_r2.max())
    #
    # _accuracy_training = "Training Accuracy: %0.4f (+/- %0.3f)" % (
    #     scores_training_data.mean(), scores_training_data.std() * 2)
    #
    # if _exclude_price_tuning and _exclude_price_trans:
    #     _mean_abs = scores_abs.mean()
    #     _std_abs = scores_abs.std()
    # elif not _exclude_price_tuning and _exclude_price_trans:
    #     _mean_abs = np.multiply(np.array(scores_abs.mean()), 1000.0)
    #     _std_abs = np.multiply(np.array(scores_abs.std()), 1000.0)
    # elif not _exclude_price_trans:
    #     _mean_abs = np.multiply(np.array(power_transform_price._scaler.mean_), 1000.0)
    #     _std_abs = np.multiply(np.array(power_transform_price._scaler.scale_), 1000.0)
    #
    # _mean_abs_percent_error = "\n MAE: %0.0f NOK (+/- %0f)\n R2: %0.3f (+/- %0.2f)" % (
    #     _mean_abs, _std_abs, scores_r2.mean(), scores_r2.std())
    #
    # plot_price_regression(predicted_prices, seen_prices, _accuracy)
    #
    # # dummy_predict = dummy_estimator.predict(x)
    #
    # # dummy_predict = power_transform_price.inverse_transform(np.array(dummy_predict).reshape(-1, 1))
    # # dummy_predict = np.array(dummy_predict).flatten()
    #
    # # plot_price_regression(dummy_predict, seen_prices, _accuracy_dummy)
    #
    # print(_accuracy)
    # print(_accuracy_mse)
    # print(_accuracy_variance)
    #
    # print(_accuracy_max)
    # print(_accuracy_training)
    #
    # save_accuracy_log(scores_r2, estimator)
    #
    #
    # def describe_split():
    #     # Data to plot
    #     labels = ['Training Set, n=' + str(len(X)), 'Testing Set, n=' + str(len(x))]
    #     sizes = [len(X), len(x)]
    #     colors = ['lightgreen', 'lightskyblue']
    #
    #     # Plot
    #     plt.pie(sizes, labels=labels, colors=colors,
    #             autopct='%1.1f%%', shadow=True, startangle=140)
    #
    #     plt.axis('equal')
    #     plt.title('Ratio of Split')
    #     plt.show()
    #
    #
    # describe_split()
    # plt.show()
    #
    # plot_dist(x=data_orig['price'], title='Original Price Distribution' + str(_mean_abs_percent_error))
    #
    # plot_dist(x=Y, title='New Price Distribution' + str(_mean_abs_percent_error))
    #
    # feat_importances = pd.Series(model.feature_importances_, index=data.columns)
    # feat_importances.nlargest(6).plot(kind='barh')
    #
    # plt.show()
    #
    # print(model.get_params())
    # print(x.columns)
    # print(model.feature_importances_)
    #
    # print(data.info(verbose=True))
    #
    # print(data.corr())
    # if not _exclude_price_trans:
    #     try:
    #         full_data = data
    #         full_data['price'] = power_transform_price.inverse_transform(np.array(label).reshape(-1, 1))
    #         sample = full_data.sample(999)
    #         sample.to_csv('../sample.csv')
    #     except:
    #         raise Exception
    #
    # d['x']['p'] = predicted_prices
    #
    # print(d['x'])

    # run(path, X_train, Y_train, X_test, Y_test)
# if _alg in 'neural':
#     from keras.models import Sequential
#     from keras.layers import Dense
#     import keras as ks
#     from keras import backend as K
#     from sklearn.model_selection import cross_val_score
#     from sklearn.model_selection import KFold
#     from keras import callbacks
#     import talos as ta
#     from sklearn.preprocessing import StandardScaler
#     # define base model
#     from numpy.random import seed
#     import tensorflow as tf
#
#     tf.Session(config=tf.ConfigProto(log_device_placement=True))
#     seed(42)
#     from tensorflow import set_random_seed
#
#     set_random_seed(42)
#
#
#     class MyLogger(ks.callbacks.Callback):
#         def __init__(self, n):
#             super().__init__()
#             self.n = n  # print loss & acc every n epochs
#
#         def on_epoch_end(self, epoch, logs={}):
#             if epoch % self.n == 0:
#                 curr_loss = logs.get('val_loss')
#
#                 print("epoch = %4d  loss = %0.5f" \
#                       % (epoch, curr_loss))
#
#
#     def optimize():
#
#         p = {'activation': [K.relu],
#              'optimizer': ['nadam'],
#              'losses': ['mean_absolute_error'],
#              'hidden_layers': [4],
#              'batch_size': [16],
#              'epochs': [100],
#              'layer_size': [23]}
#
#         def price_model(x_train, y_train, x_val, y_val, params):
#             my_logger = MyLogger(n=1)
#
#             print(x_train.head())
#             model = Sequential()
#
#             model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], activation=params['activation']))
#             model.add(Dropout(0.33))
#             model.add(Dense(8, activation=params['activation']))
#             # model.add(Dropout(0.33))
#             model.add(Dense(1, activation='relu'))
#             model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['mean_absolute_error'])
#
#             out = model.fit(x_train, y_train,
#                             batch_size=params['batch_size'],
#                             epochs=params['epochs'],
#                             validation_data=[x_val, y_val],
#                             callbacks=[my_logger],
#                             verbose=1)
#
#             print(d['x'].head())
#
#             y_predict = model.predict(d['x'])
#             print(np.mean(np.abs(y_predict - y.values)))
#             p_data = pd.DataFrame(data=y_predict, columns=['price'])
#             abs = p_data
#
#             y_rev = np.array(d['y']).reshape(-1, 1)
#
#             print(np.mean(y_rev - abs))
#
#             plt.ylim(bottom=15, top=35)
#
#             plt.plot(out.history['loss'])
#             plt.plot(out.history['val_loss'])
#             plt.title('Mean Absolute Error: %.2f' % (np.median(out.history['val_loss'])))
#
#             # print("Results: min: %.2f max: %.2f MAE" % (_mean_abs.min(), _mean_abs.max()))
#             plt.ylabel('Loss')
#             plt.xlabel('Epoch')
#             plt.legend(['Train', 'Test'], loc='upper left')
#             plt.show()
#
#             return out, model
#
#         scan_object = ta.Scan(x=X, y=Y.values, x_val=x, y_val=y.values, model=price_model, params=p, print_params=True)
#
#         return scan_object
#
#
#     e = ta.Evaluate(optimize())

    # e.evaluate(X, Y.values, mode='regression', print_out=True, metric='mean_absolute_error', asc=True)

    # fix random seed for reproducibility

    # evaluate model with standardized dataset
    # print(Y.head())
    # print(y.head())
    # y_predict = model.predict(x)

    # p_data = pd.DataFrame(data=y_predict, columns=['price'])
    # abs = np.multiply(power_transform_price.inverse_transform(p_data), 1000.0)

    # y_rev = np.multiply(power_transform_price.inverse_transform(np.array(d['y']).reshape(-1, 1)), 1000.0)

    # print(np.mean(y_rev - abs))

# kfold = KFold(n_splits=2, random_state=seed)
# results = cross_val_score(estimator, x, y, cv=kfold, scoring='neg_mean_absolute_error')

# print(results.mean())
# mean = results.mean()

# print("Results: min: %.2f max: %.2f MAE" % (_mean_abs.min(), _mean_abs.max()))


# SVR

"""
parameters = {'kernel':('linear', 'rbf'), 'C':[1.5, 10], 'gamma':[1e-7, 1e-4], 'epsilon':[0.1,0.2,0.3,0.5]}
svr = svm.SVR()
clf = GridSearchCV(svr, parameters, verbose=2, n_jobs=-1)
clf.fit(X, Y)
clf.best_params_
"""


# parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10], 'gamma':[1e-7, 1e-4, 0.001, 0.1], 'epsilon':[0.3,0.4]}

# parameters = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]},

#             {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10], 'epsilon': [0.1, 0.3, 0.4]}]

# parameters = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]},
#                 {'kernel': ['poly'], 'C': [0.01, 0.1, 1, 10, 100], 'epsilon': [0.001, 0.01, 0.1],
#                 'degree': [3], 'coef0': [1]}]

# clf = GridSearchCV(svr, parameters, verbose=2, n_jobs=-1)
# clf.fit(X, Y)
# print(_alg)
# print(clf.best_params_)
# best_params = clf.best_params_

# use best_params_
# if best_params['kernel'] == 'rbf':
#    regressor = svm.SVR(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
# else:
#    regressor = svm.SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])
# regressor = svm.SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
# regressor.fit(X, Y)

# regressor = svm.SVR(kernel='rbf', C=10, epsilon=0.3, gamma=0.1)
# regressor.fit(X, Y)

# better

# regressor = svm.SVR(kernel='linear', C=1.5, epsilon=0.3, gamma=1e-07)
# regressor.fit(X, Y)


# BayesianRidge

# parameters_bayes = {'alpha_1': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04], 'alpha_2': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04],
#                   'lambda_1': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04],
#                   'lambda_2': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04], 'compute_score': [True]}

# bayes = GridSearchCV(br, parameters_bayes, verbose=2, n_jobs=-1)
# bayes.fit(X, Y)
# print(_alg)
# print(bayes.best_params_)
# best_params = bayes.best_params_

# use best_params_
# regressor = BayesianRidge(compute_score=True, alpha_1=best_params['alpha_1'], alpha_2=best_params['alpha_2'],
# lambda_1=best_params['lambda_1'], lambda_2=best_params['lambda_2'])
# regressor.fit(X, Y)


# BayesianRidge

# parameters_bayes = {'alpha_1': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04], 'alpha_2': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04],
#                   'lambda_1': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04],
#                   'lambda_2': [1e-08, 1e-07, 1e-06, 1e-05, 1e-04], 'compute_score': [True]}
# bayes = GridSearchCV(br, parameters_bayes, verbose=2, n_jobs=-1)
# bayes.fit(X, Y)
# print(_alg)
# print(bayes.best_params_)
# best_params = bayes.best_params_

# use best_params_
# regressor = BayesianRidge(compute_score=True, alpha_1=best_params['alpha_1'], alpha_2=best_params['alpha_2'],
# lambda_1=best_params['lambda_1'], lambda_2=best_params['lambda_2'])
# regressor.fit(X, Y)


#metrics.explained_variance_score(y_true, y_pred) 	Explained variance regression score function
#metrics.mean_absolute_error(y_true, y_pred) 	Mean absolute error regression loss
#metrics.mean_squared_error(y_true, y_pred[, …]) 	Mean squared error regression loss
#metrics.mean_squared_log_error(y_true, y_pred) 	Mean squared logarithmic error regression loss
#metrics.median_absolute_error(y_true, y_pred) 	Median absolute error regression loss
#metrics.r2_score(y_true, y_pred[, …]) 	R^2 (coefficient of determination) regression score function.

regressor_list = []
# Decision Trees
regressor_list.append(DecisionTreeRegressor(random_state=rng))
regressor_list.append(AdaBoostRegressor(DecisionTreeRegressor(random_state=rng)))


# Neighbours
regressor_list.append(KNeighborsRegressor())

# Neural Network
regressor_list.append(MLPRegressor(random_state=rng))

# Kernel Ridge
regressor_list.append(KernelRidge())

# Ensemble
regressor_list.append(BaggingRegressor(random_state=rng))
regressor_list.append(RandomForestRegressor(random_state=rng))
regressor_list.append(GradientBoostingRegressor(random_state=rng))
regressor_list.append(ExtraTreesRegressor(random_state=rng))

# Linear Models
regressor_list.append(BayesianRidge())
regressor_list.append(Lasso(random_state=rng))
def initial_estimator_comparison():
    def comparee_estimator_results_cv():
        score_list = []
        estimator_list = []

        for regressor in regressor_list:
            print("Fitting Estimator: %s" % regressor.__class__.__name__)
            regressor.fit(TR_DATA, TR_LABEL)

            scores = cross_val_score(regressor, V_DATA, V_LABEL,
                                     cv=KFold(n_splits=10, random_state=rng),
                                     n_jobs=8, scoring='r2')
            print(scores)
            score_list.append(scores)
            estimator_list.append(regressor.__class__.__name__)
        aggregate_score = np.mean(score_list)
        fig = sns.boxplot(y=estimator_list, x=score_list, order=estimator_list, width=0.8)
        plt.axvline(x=0.8, label='threshold',c='r')
        plt.legend()
        plt.show()

#
def optimize_estimators():
    print(regressor_list)
    def comparee_estimator_results_cv():
        score_list = []
        estimator_list = []

        for regressor in regressor_list:
            print("Fitting Estimator: %s" % regressor.__class__.__name__)
            print(regressor)
            regressor.fit(TR_DATA, TR_LABEL)

            scores = cross_val_score(regressor, V_DATA, V_LABEL,
                                     cv=KFold(n_splits=10, random_state=rng),
                                     n_jobs=8, scoring='r2')
            print(scores)
            score_list.append(np.round(np.mean(scores),3))
            estimator_list.append(regressor.__class__.__name__ + ' ' + str(np.round(np.mean(scores),3)))
            accuracy = "EV: %0.4f%% (+/- %0.3f)" % (scores.mean(), scores.std() * 2)
            print(accuracy)
        aggregate_score = np.mean(score_list)
        sns.set(style="whitegrid")
        sns.barplot(y=estimator_list, x=score_list)
        plt.xlabel('Aggregated median R2: %0.4f' % np.median(score_list))
        plt.axvline(x=0.8, label='threshold',c='r')

        plt.legend()
        plt.show()

    comparee_estimator_results_cv()

def absolute_error():
    def comparee_estimator_results_cv():
        score_list = []
        estimator_list = []

        for regressor in regressor_list:
            print("Fitting Estimator: %s" % regressor.__class__.__name__)
            regressor.fit(TR_DATA, TR_LABEL)


            scores = cross_val_score(regressor, V_DATA, V_LABEL,
                                     cv=KFold(n_splits=3, random_state=rng),
                                     n_jobs=8, scoring='neg_median_absolute_error')
            print(scores)
            score_list.append(abs(np.round(np.mean(scores),1)))
            estimator_list.append(regressor.__class__.__name__ + ' ' + str(abs(np.round(np.median(scores),1))))
        sns.set(style="whitegrid")
        sns.barplot(y=estimator_list, x=score_list)
        plt.xlabel('Avg. Absolute Median Error in 1000 NOK: %0.4f' % np.mean(score_list))
        plt.legend()
        plt.show()

    comparee_estimator_results_cv()
absolute_error()
print('************************LAST REPORT***********************************')

print('************************LAST REPORT***********************************')

#initial_estimator_comparison()
