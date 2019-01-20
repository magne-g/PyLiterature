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

#data = data.sample(500)

print(data.describe())
print(data.count())
print(data.corr())


g = sns.boxplot('manufacturer', 'price', data=data)

locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

plt.show()
