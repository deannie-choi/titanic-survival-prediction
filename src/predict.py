import os
import sys
import joblib
import pandas as pd

# Ensure the 'src' folder is in the Python path
BASE_DIR = os.path.abspath("/Users/dean/Projects/titanic-survival-prediction")
SRC_DIR = os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/src")
sys.path.append(SRC_DIR)

from preprocess import preprocess_data

# Load test dataset
test_df = pd.read_csv(os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/datasets", "test.csv"))

# Preprocess test data
test_df = preprocess_data(test_df)

# Load trained model
model_path = os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/models", "random_forest.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

rf_model = joblib.load(model_path)

# Predict survival
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
X_test = test_df[features]
test_predictions = rf_model.predict(X_test)

# Save submission file
SUBMISSION_DIR = os.path.join(BASE_DIR, "/Users/dean/Projects/titanic-survival-prediction/submissions")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

submission_file = os.path.join(SUBMISSION_DIR, "/Users/dean/Projects/titanic-survival-prediction/submission_final.csv")
pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_predictions}).to_csv(submission_file, index=False)

print(f"✅ Submission file created: {submission_file}")
