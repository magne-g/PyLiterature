# ML-Project1
ITI43210 Machine Learning Project 1 - Classification and regression trees

## Introduction

We aim at solving the problem of predicting pricing on used cars on the norwegian market. To do this we have gathered some preliminary raw data from a popular norwegian sales platform for used cars. We collected all available attributes that were more or less common amongst each car; such as:

- Price
- Model
- Year of production
- Mileage
- Color
- RWD, FWD or 4WD
- First date of registration
- Fuel type
- Transmission type
- Metric horsepower

The raw data needs to be analyzed and cleaned before we consider how to best transform the features into usable input for the decision tree algorithm.

Cleaning the data involves methods such as:

- Remove duplicate entries
- Properly format continous values such as price and mileage.
- Attempt to sanitize and categorize deviating attribute values

##Preprocessing

During the first stage of preprocessing, we had the goal of formatting the raw data to prepare it for computational analysis while still preserving human readability. We used python and pandas to preprocess the data.

###Steps

- Designate finn_code as index (unique identifier)

- Remove duplicate rows

- Truncate varying transmission layouts into FWD, RWD, and AWD (All-Wheel-Drive)

- Truncate varying fuel types into Petrol, Diesel, Electricity, Hybrid, and Hybrid biofuel

- Properly format price values as a numerical value

- Properly format metric horse power values as numerical

- Properly format mileage values as numerical

- Remove day and month in the date for first time registration

- Properly format engine size as a numerical value

- Separate car manufacturer from model

- Exclude lesser known car manufacturers based on the following filter:
```
include_manufacturers = ['Audi', 'BMW', 'Citroen', 'Fiat', 'Ford', 'Hyundai', 'Kia', 'Mazda', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen', 'Volvo', 'Tesla']
```

- Drop the color column for now

- Reorder columns for human readability

##Analysis

Total non-null values:

```
finn_code       10564
model_year      10564
manufacturer    10564
model           10533
km              10562
power           10438
gear            10556
trans           10564
first_reg       9528 
cylinder        9373 
fuel_type       10564
price           10564
```

Statistical data:

```
          finn_code    model_year            km         power    first_reg     cylinder         price
count  1.056400e+04  10564.000000  1.056200e+04  10438.000000  9528.000000  9373.000000  1.056400e+04
mean   1.365265e+08  2012.150133   9.979486e+04  149.064955    2012.258816  220.539891   2.243525e+05
std    6.324521e+06  5.875333      8.687093e+04  68.486634     5.360243     648.775393   1.928690e+05
min    5.525575e+07  1932.000000   0.000000e+00  0.000000      1922.000000  0.000000     1.286000e+03
25%    1.377989e+08  2009.000000   3.530000e+04  109.000000    2009.000000  1.600000     7.879375e+04
50%    1.380084e+08  2014.000000   7.500000e+04  136.000000    2014.000000  2.000000     1.799000e+05
75%    1.381339e+08  2016.000000   1.500000e+05  170.000000    2016.000000  2.300000     3.150000e+05
max    1.382291e+08  2019.000000   1.711105e+06  800.000000    2104.000000  7272.000000  2.169000e+06
```

Correlation coefficients:

```
            finn_code  model_year        km     power  first_reg  cylinder     price
finn_code   1.000000   0.142982   -0.145551 -0.018425  0.131777  -0.025357  0.089757
model_year  0.142982   1.000000   -0.708934  0.193384  0.925558  -0.278476  0.570921
km         -0.145551  -0.708934    1.000000 -0.114931 -0.709616   0.275173 -0.503956
power      -0.018425   0.193384   -0.114931  1.000000  0.202303   0.001741  0.658121
first_reg   0.131777   0.925558   -0.709616  0.202303  1.000000  -0.276275  0.578710
cylinder   -0.025357  -0.278476    0.275173  0.001741 -0.276275   1.000000 -0.170694
price       0.089757   0.570921   -0.503956  0.658121  0.578710  -0.170694  1.000000
```

Boxplot manufacturer/price:


###Discussion

In the statistical data we can observe max values that would qualify as extreme outliers, such as 'cylinder' which specifies engine size in liters. The high mean may indicate that this is not the only outlier and may be instances where volume is represented in milliliters instead of liters. Further preprocessing is needed here.

Further investigation is needed to determine if a min value of 0 is the same as a null-value.

Correlation data shows:
 
 - Moderate positive correlation between price and production year
 - Moderate negative correlation between price and mileage
 - Moderate/strong positive correlation between price and metric horse power
 - Strong negative correlation between production year and mileage
 
 

