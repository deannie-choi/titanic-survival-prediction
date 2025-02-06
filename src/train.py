import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure the 'src' folder is in the Python path
BASE_DIR = os.path.abspath("/Users/dean/Projects/titanic-survival-prediction")
SRC_DIR = os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/src")
sys.path.append(SRC_DIR)

from preprocess import preprocess_data

# Load dataset
train_df = pd.read_csv(os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/datasets", "train.csv"))

# Preprocess data
train_df = preprocess_data(train_df)

# Train model
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
X = train_df[features]
y = train_df["Survived"]

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# Save model
os.makedirs(os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/models"), exist_ok=True)
joblib.dump(rf_model, os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/models", "random_forest.pkl"))
print("✅ Model saved as models/random_forest.pkl")