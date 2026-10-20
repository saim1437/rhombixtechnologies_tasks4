# House Price Prediction Model
# Libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading
data = pd.read_csv("HousingData.csv")
print(data.head())

# data exploring
data.info()
print("\n")
print(data.describe())

# data cleaning
print("Missing values:\n", data.isnull().sum())
data = data.drop_duplicates()
# missing values are filled with the median later (inside the pipeline),
# so we do not lose any rows

# MEDV = median house price in $1000s
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# quick look: how each feature is related to price
print("\nCorrelation with MEDV:")
print(data.corr()['MEDV'].sort_values(ascending=False))

# model train and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# helper to print the evaluation of a model
def evaluate(name, model):
    pred = model.predict(X_test)
    print(f"\n{name}:")
    print("MAE: ", round(mean_absolute_error(y_test, pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
    print("R2:  ", round(r2_score(y_test, pred), 4))
    return r2_score(y_test, pred)

# each pipeline = fill missing values -> normalize -> model
def make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Model 1: Linear Regression
model_lr = make_pipeline(LinearRegression())
model_lr.fit(X_train, y_train)
r2_lr = evaluate("Linear Regression", model_lr)

# Model 2: Decision Tree
model_dt = make_pipeline(DecisionTreeRegressor(random_state=42))
model_dt.fit(X_train, y_train)
r2_dt = evaluate("Decision Tree", model_dt)

# Model 3: Random Forest
model_rf = make_pipeline(RandomForestRegressor(random_state=42))
model_rf.fit(X_train, y_train)
r2_rf = evaluate("Random Forest", model_rf)

# Model 4: Gradient Boosting (same idea as XGBoost)
model_gb = make_pipeline(GradientBoostingRegressor(random_state=42))
model_gb.fit(X_train, y_train)
r2_gb = evaluate("Gradient Boosting", model_gb)

# hyperparameter tuning (GridSearchCV with 5-fold cross validation)
print("\nTuning Random Forest...")
rf_params = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(make_pipeline(RandomForestRegressor(random_state=42)),
                       rf_params, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Best parameters:", grid_rf.best_params_)
print("Best CV R2:", round(grid_rf.best_score_, 4))
r2_rf_tuned = evaluate("Random Forest (tuned)", grid_rf.best_estimator_)

print("\nTuning Gradient Boosting...")
gb_params = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [2, 3, 4],
    'model__subsample': [0.8, 1.0]
}
grid_gb = GridSearchCV(make_pipeline(GradientBoostingRegressor(random_state=42)),
                       gb_params, cv=5, scoring='r2', n_jobs=-1)
grid_gb.fit(X_train, y_train)
print("Best parameters:", grid_gb.best_params_)
print("Best CV R2:", round(grid_gb.best_score_, 4))
r2_gb_tuned = evaluate("Gradient Boosting (tuned)", grid_gb.best_estimator_)

# compare all models and pick the best one
results = {
    'Linear Regression': (r2_lr, model_lr),
    'Decision Tree': (r2_dt, model_dt),
    'Random Forest': (r2_rf, model_rf),
    'Gradient Boosting': (r2_gb, model_gb),
    'Random Forest (tuned)': (r2_rf_tuned, grid_rf.best_estimator_),
    'Gradient Boosting (tuned)': (r2_gb_tuned, grid_gb.best_estimator_),
}
print("\nModel comparison (R2 on test set):")
for name, (score, _) in sorted(results.items(), key=lambda item: item[1][0], reverse=True):
    print(f"{name}: {round(score, 4)}")

best_name = max(results, key=lambda k: results[k][0])
best_model = results[best_name][1]
print("\nBest model:", best_name)

# plots: actual vs predicted, and feature importance
pred_best = best_model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_best, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual price ($1000s)")
plt.ylabel("Predicted price ($1000s)")
plt.title(f"Actual vs Predicted ({best_name})")
plt.savefig("actual_vs_predicted.png", dpi=120, bbox_inches='tight')
plt.close()

final_model = best_model.named_steps['model']
if hasattr(final_model, 'feature_importances_'):
    importance = pd.Series(final_model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 5))
    importance.plot(kind='barh')
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("\nFeature importance:")
    print(importance.sort_values(ascending=False).round(4))

# Command-line interface
# Only the main features are asked. The rest use the median of the dataset.
main_features = {
    'RM': 'Average number of rooms per house',
    'LSTAT': '% of lower status population in the area',
    'PTRATIO': 'Pupil-teacher ratio in the area',
    'CRIM': 'Crime rate per capita',
    'AGE': '% of houses built before 1940 (age of property)',
    'DIS': 'Distance to employment centres',
}

def predict_price(user_values):
    row = X.median().to_dict()      # start with median for every feature
    row.update(user_values)         # replace with what the user entered
    row_df = pd.DataFrame([row])[X.columns]
    return best_model.predict(row_df)[0]

# example prediction
example = {'RM': 6.5, 'LSTAT': 10, 'PTRATIO': 18, 'CRIM': 0.3, 'AGE': 60, 'DIS': 4}
print("\nExample house:", example)
print(f"Predicted price: ${predict_price(example) * 1000:,.0f}")

if __name__ == "__main__":
    while True:
        choice = input("\nPredict a house price? (y/n): ").strip().lower()
        if choice != 'y':
            break
        values = {}
        for col, text in main_features.items():
            while True:
                try:
                    values[col] = float(input(f"{text} [{col}]: "))
                    break
                except ValueError:
                    print("Please enter a number.")
        print(f"Predicted price: ${predict_price(values) * 1000:,.0f}")
