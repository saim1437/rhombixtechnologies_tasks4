# House Price Prediction Model

## Overview
This project predicts the selling price of a house using features like number of rooms, crime rate, age of the property, and distance to employment centres. It trains several regression models, tunes the best ones, and lets the user get a price prediction from the command line.

## Dataset
`HousingData.csv` — Boston Housing dataset (506 houses, 13 features). The target column is `MEDV` (median house price in $1000s). Some columns (`CRIM`, `ZN`, `INDUS`, `CHAS`, `AGE`, `LSTAT`) have missing values.

| Column | Meaning |
|---|---|
| CRIM | Crime rate per capita |
| ZN | % of residential land zoned for large lots |
| INDUS | % of non-retail business acres |
| CHAS | Borders Charles River (1) or not (0) |
| NOX | Air pollution (nitric oxide) level |
| RM | Average number of rooms |
| AGE | % of houses built before 1940 (age of property) |
| DIS | Distance to employment centres |
| RAD | Access to highways index |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Demographic index |
| LSTAT | % of lower status population |
| MEDV | Median house price ($1000s) — target |

## How It Works
1. Load and explore the data.
2. Clean the data (remove duplicates, fill missing values with the median).
3. Check how each feature is correlated with the price.
4. Normalize features and split into train (80%) and test (20%) sets.
5. Train and compare 4 models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
6. Improve the Random Forest and Gradient Boosting models with hyperparameter tuning (GridSearchCV, 5-fold cross validation).
7. Evaluate each model using MAE, RMSE, and R2 score.
8. Pick the best model, save the plots, and show feature importance.
9. Predict the price of a new house through the command-line interface.

## Results
| Model | R2 (test set) |
|---|---|
| Gradient Boosting (tuned) | 0.9014 |
| Gradient Boosting | 0.9004 |
| Random Forest | 0.8887 |
| Random Forest (tuned) | 0.8858 |
| Linear Regression | 0.6593 |
| Decision Tree | 0.6436 |

The tuned Gradient Boosting model is the best one (MAE about 1.96, meaning the average error is about $1,960). The most important features are `RM` (rooms) and `LSTAT` (lower status %).

Note: the test set is small (about 100 houses), so the scores of the top models are very close to each other.

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
```
Install with:
```
pip install pandas numpy scikit-learn matplotlib
```

## How to Run
Keep `HousingData.csv` in the same folder, then:
```
python house_price_prediction.py
```
Or open `house_price_prediction.ipynb` in Jupyter Notebook / Google Colab.

## Command-Line Interface
After training, the script asks for the main features of a house. The other features use the median value of the dataset.
```
Predict a house price? (y/n): y
Average number of rooms per house [RM]: 6.5
% of lower status population in the area [LSTAT]: 10
Pupil-teacher ratio in the area [PTRATIO]: 18
Crime rate per capita [CRIM]: 0.3
% of houses built before 1940 (age of property) [AGE]: 60
Distance to employment centres [DIS]: 4
Predicted price: $22,473
```

## Output
- Metrics (MAE, RMSE, R2) for every model
- Best hyperparameters from GridSearchCV
- `actual_vs_predicted.png` — predicted vs actual prices
- `feature_importance.png` — which features affect the price most
- Predicted price for a new house
# rhombixtechnologies_tasks
