# ML-Project1
ITI43210 Machine Learning Project 1 - Classification and regression trees

- Describe the problem to be solved and your dataset. How was the raw data generated? What applications does the problem have? What predictors, that is attributes, are there and how are they converted to suitable input for your chosen tree learning algorithms?

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
- Attempt to sanetize and standardize deviating attribute values

Transformation of data involves methods such as:

- Bucketing



Software to explore

- XGBoost
- C5.0
- RandomForest

