import pandas as pd
import numpy as np
import sys
import csv


def pp_trans(dataframe):
    trans = dataframe['trans']
    trans = trans.str.replace('.*(x(|-| |)drive|quat|4|awd|sperre|awc|fire).*', 'AWD', regex=True, case=False)
    trans = trans.str.replace('.*(forhjul|fwd).*', 'FWD', regex=True, case=False)
    trans = trans.str.replace('.*(bakhjul|rwd).*', 'RWD', regex=True, case=False)
    dataframe['trans'] = trans
    filter = ['AWD', 'FWD', 'RWD']
    dataframe = dataframe[dataframe.trans.str.contains('|'.join(filter), na=False)]

    return dataframe

def pp_price(dataframe):
    price = dataframe['price']
    price = price.str.replace('(,-| +)', '', regex=True)
    dataframe['price'] = price
    return dataframe

def pp_power(dataframe):
    power = dataframe['power']
    power = power.str.replace('(hk| +)', '', regex=True, case=False)
    dataframe['power'] = power
    return dataframe

def pp_km(dataframe):
    km = dataframe['km']
    km = km.str.replace('(km| +)', '', regex=True, case=False)
    dataframe['km'] = km
    return dataframe


if len(sys.argv) != 2 or '.csv' not in sys.argv[1]:
    sys.exit("Usage: clean.py filename.csv")

f = sys.argv[1]

df = pd.read_csv(f)
df = df.drop_duplicates()

df = df.set_index('finn_code')

df = pp_trans(df)
df = pp_price(df)
df = pp_power(df)
df = pp_km(df)

print(df['km'].unique())
print(len(df))

df.to_csv('processed_' + f)










