import pandas as pd
import numpy as np
import sys
import csv


def pp_trans(dataframe): #Truncates (FWD, RWD, AWD) transmission layouts
    trans = dataframe['trans']

    trans = trans.str.replace('.*(x(|-| |)drive|quat|4|awd|sperre|awc|fire).*', 'AWD', regex=True, case=False)
    trans = trans.str.replace('.*(forhjul|fwd).*', 'FWD', regex=True, case=False)
    trans = trans.str.replace('.*(bakhjul|rwd).*', 'RWD', regex=True, case=False)
    dataframe['trans'] = trans
    filter = ['AWD', 'FWD', 'RWD']
    dataframe = dataframe[dataframe.trans.str.contains('|'.join(filter), na=False)]

    return dataframe

def pp_fuel_type(dataframe): #Truncates fuel types
    fuel_type = dataframe['fuel_type']

    fuel_type = fuel_type.str.replace('elektrisitet\+.*', 'Hybrid', regex=True, case=False)
    fuel_type = fuel_type.str.replace('gass\+.*', 'Hybrid Bio', regex=True, case=False)
    dataframe['fuel_type'] = fuel_type
    filter = ['Bensin', 'Diesel', 'Elektrisitet', 'Hybrid', 'Hybrid Bio', 'Gass']
    dataframe = dataframe[dataframe.fuel_type.str.contains('|'.join(filter), na=False)]

    return dataframe

def pp_price(dataframe): #Converts price values to integers
    price = dataframe['price']

    price = price.str.replace('(,-| +)', '', regex=True)
    price = pd.to_numeric(price, errors='raise')

    dataframe['price'] = price
    return dataframe

def pp_power(dataframe): #Converts metric horse power to floats and NaN
    power = dataframe['power']

    power = power.str.replace('(hk| +)', '', regex=True, case=False)
    power = pd.to_numeric(power, errors='raise').astype(int)

    dataframe['power'] = power
    return dataframe

def pp_km(dataframe): #Converts mileage to floats and NaN
    km = dataframe['km']

    km = km.str.replace('(km| +)', '', regex=True, case=False)
    km = pd.to_numeric(km, errors='raise')

    dataframe['km'] = km
    return dataframe

def pp_first_reg(dataframe): #Formats dates, removes day and month

    dataframe['first_reg'] = pd.to_datetime(df['first_reg'], errors='coerce', exact=True)
    dataframe['first_reg'] = dataframe.first_reg.dt.to_period('Y')

    return dataframe

def pp_cylinder(dataframe):

    cylinder = dataframe['cylinder']

    cylinder = cylinder.replace(',', '.', regex=True).replace('l| +', '', regex=True)

    dataframe['cylinder'] = pd.to_numeric(cylinder, errors='coerce')
    return dataframe

def pp_model(dataframe): #Seperate manufacturer from model

    model = dataframe['model']

    manufacturers = ['Audi', 'BMW', 'Citroen', 'Fiat', 'Ford', 'Hyundai', 'Kia', 'Mazda', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugot', 'Renault', 'Skoda', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen', 'Volvo']






if len(sys.argv) != 2 or '.csv' not in sys.argv[1]:
    sys.exit("Usage: clean.py filename.csv")

f = sys.argv[1]

df = pd.read_csv(f)
df = df.drop_duplicates()
df = df.set_index('finn_code')

df = pp_trans(df)
df = pp_fuel_type(df)
df = pp_price(df)
df = pp_power(df)
df = pp_km(df)
df = pp_first_reg(df)
df = pp_cylinder(df)

#print(df['first_reg'].unique())
print(len(df))

df.to_csv('processed_' + f)










