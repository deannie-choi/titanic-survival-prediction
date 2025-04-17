# Titanic - Machine Learning from Disaster

This project tackles the classic Kaggle competition "Titanic: Machine Learning from Disaster."  
The goal is to predict whether a passenger survived or not based on attributes such as class, gender, age, and fare.

## 🔍 Project Objective

To develop a classification model that predicts survival outcomes of Titanic passengers using data preprocessing, exploratory data analysis (EDA), and XGBoost classification.

## 📁 Project Structure

    titanic-survival-prediction/
    ├── Data/                  # Raw data files (train.csv, test.csv)
    ├── notebooks/             # Jupyter Notebooks for EDA and modeling
    │   ├── 01_eda_preprocessing.ipynb
    │   └── 02_model_training.ipynb
    ├── models/                # Saved model files (optional)
    ├── outputs/               # submission.csv for Kaggle
    ├── requirements.txt       # Python dependencies
    └── README.md              # Project documentation

## 📊 Features Used

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

All features were cleaned and encoded appropriately prior to modeling.

## ⚙️ Model

- **XGBoost Classifier**
  - `use_label_encoder=False`
  - `eval_metric="logloss"`
  - Random state: 42
- Performance evaluated on a validation split (80/20)

## 🧪 Validation Result
Accuracy: ~0.81 (depending on split)

## 📤 Submission

- A `submission.csv` file was generated using the final model.
- File located at: `outputs/submission.csv`
- Compatible with Kaggle format: `PassengerId`, `Survived`

## 🚀 How to Run

1. Clone the repository  
   `git clone https://github.com/deannie-choi/titanic-survival-prediction.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Run notebooks inside `notebooks/` in order:
   - `01_eda_preprocessing.ipynb`
   - `02_model_training.ipynb`

## 📎 Resources

- Kaggle Competition: [https://www.kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
- Dataset: Provided directly from Kaggle competition page

## 📝 Author

- **Dean Choi**
- Graduate Student in Data Science  
- [LinkedIn Profile](https://www.linkedin.com/in/dean-choi/)