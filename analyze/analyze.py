import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_info_columns', 500)

data = pd.read_csv('../processed_cars.csv')

#data = data.dropna()

data = data.sample(100)

#print(data.describe())
#print(data.count())
#print(data.corr())

data = data[data['price'] < 750000]
data = data[data['price'] > 5000]

data = data[data['km'] < 400000]




def box_plot():
    g = sns.boxplot('manufacturer', 'price', data=data)


sns.relplot('price', 'km', data=data, hue='fuel_type')

locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

plt.show()
