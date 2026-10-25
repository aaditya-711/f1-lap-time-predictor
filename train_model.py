
"""
F1 Lap Time Prediction Model Training Script
Uses sample_data.csv instead of generating data
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("F1 LAP TIME PREDICTION MODEL - TRAINING SCRIPT")
print("=" * 70)

np.random.seed(42)


print("\nLoading dataset from CSV file...")

df = pd.read_csv("sample_data.csv")

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())


print("\n" + "=" * 70)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

label_encoders = {}
categorical_cols = ['Driver', 'Compound', 'Weather', 'TrackStatus']
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {df[col].nunique()} unique values")


feature_columns = [
    'LapNumber', 'TyreLife', 'TrackTemp', 'AirTemp', 'FuelLoad',
    'Position', 'DRS_Available', 'FreshTyres',
    'Compound_encoded', 'Weather_encoded', 'TrackStatus_encoded', 'Driver_encoded'
]

print("\n" + "=" * 70)
print("SELECTED FEATURES (12 FEATURES TO AVOID OVERFITTING)")
print("=" * 70)
for i, feature in enumerate(feature_columns, 1):
    print(f"{i:2d}. {feature}")

X = df_encoded[feature_columns]
y = df_encoded['LapTime']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set:  {X_test.shape[0]} samples")

print("\n" + "=" * 70)
print("TRAINING XGBOOST MODEL")
print("=" * 70)

model = xgb.XGBRegressor(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train, verbose=False)
print("✓ Training complete!")


y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

train_mae  = mean_absolute_error(y_train, y_train_pred)
test_mae   = mean_absolute_error(y_test,  y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test,  y_test_pred)

print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15} {'Difference':<15}")
print("-" * 65)
print(f"{'MAE (seconds)':<20} {train_mae:<15.3f} {test_mae:<15.3f} {abs(train_mae-test_mae):<15.3f}")
print(f"{'RMSE (seconds)':<20} {train_rmse:<15.3f} {test_rmse:<15.3f} {abs(train_rmse-test_rmse):<15.3f}")
print(f"{'R² Score':<20} {train_r2:<15.3f} {test_r2:<15.3f} {abs(train_r2-test_r2):<15.3f}")

gap = train_r2 - test_r2
status_msg = "WELL-GENERALIZED ✓" if gap < 0.05 else ("ACCEPTABLE ✓" if gap < 0.10 else "OVERFITTING ⚠")
print(f"\nOverfitting Gap: {gap:.3f}  →  {status_msg}")


fi = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:\n", fi.to_string(index=False))


print("\n" + "=" * 70)
print("SAVING MODEL FILES")
print("=" * 70)

with open('f1_laptime_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ f1_laptime_model.pkl")

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ label_encoders.pkl")

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ feature_columns.pkl")

print("\n" + "=" * 70)
print("ALL DONE!  →  Run: python -m streamlit run f1_app.py")
print("=" * 70)