# Titanic Survival Prediction 🚢

This project is a complete end-to-end machine learning case study using the Titanic dataset. It includes detailed data exploration, preprocessing, model building, evaluation, tuning, and final conclusions, all aligned with practical ML workflow standards. The purpose is to predict survival of passengers based on Titanic manifest features using multiple classification models.

---

## 📊 1. Project Objectives

- Understand which features most impact passenger survival.
- Explore survival patterns via EDA and visualizations.
- Clean and transform the data using best preprocessing practices.
- Apply and compare baseline and ensemble classifiers.
- Use evaluation metrics, cross-validation, and hyperparameter tuning.
- Save the best model for reproducibility.

---

## 📁 2. Project Structure

| File | Description |
|------|-------------|
| `01_EDA.ipynb` | Exploratory data analysis: survival rate trends and correlations |
| `02_Preprocessing.ipynb` | Data cleaning, feature encoding, and feature engineering |
| `03_Model_Logistic.ipynb` | Baseline Logistic Regression training and evaluation |
| `04_Model_RF_XGB.ipynb` | Ensemble models (Random Forest, XGBoost) implementation |
| `05_Tuning_Validation.ipynb` | Cross-validation, hyperparameter search, evaluation & model saving |
| `best_random_forest_model.pkl` | Serialized final model using `joblib` |

---

## 🔍 3. Exploratory Data Analysis (EDA)

EDA was used to examine patterns in survival rate.

### 🧭 Key Findings:
- Female passengers had a higher survival rate than males.
- 1st class passengers had significantly better survival chances.
- Children (under 10) showed higher survival rates.

### 📊 Visual Samples

```python
sns.barplot(x="Pclass", y="Survived", data=df)
sns.barplot(x="Sex", y="Survived", data=df)
```

Plots:
- `pclass_survival.png` – Survival by class
- `sex_survival.png` – Survival by gender

---

## 🛠️ 4. Preprocessing Pipeline

- Missing `Age`, `Fare`: filled with median.
- `Embarked`: filled with mode.
- Dropped: `Name`, `Ticket`, `Cabin`, `PassengerId`.
- Encoded categorical vars using `map()` and `get_dummies()`.
- Created:
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone = (FamilySize == 1).astype(int)`

---

## 🤖 5. Model Training & Performance

### 🧪 Models Trained:
- Logistic Regression (baseline)
- Random Forest
- XGBoost Classifier

### 📈 Accuracy Comparison:
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 79.3% |
| XGBoost              | 81.0% |
| Random Forest (tuned) | **83.3%** ✅ |

---

## 🔎 6. Evaluation Metrics

### Confusion Matrix (Random Forest)
```python
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
```
<p align="center">
  <img src="https://raw.githubusercontent.com/your-username/titanic-survival-prediction/main/images/confusion_matrix.png" alt="Confusion Matrix" width="500"/>
</p>

### Classification Report:
```
              precision    recall  f1-score   support
           0       0.84      0.89      0.87       105
           1       0.81      0.74      0.77        74

    accuracy                           0.83       179
   macro avg       0.82      0.82      0.82       179
weighted avg       0.83      0.83      0.83       179
```

---

## 🔧 7. Hyperparameter Tuning

### RandomizedSearchCV (Random Forest)
```python
param_grid = {
  'n_estimators': [100, 200, 300],
  'max_depth': [None, 5, 10],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4]
}
```

### Best Result:
| Parameter | Value |
|-----------|--------|
| `n_estimators` | 100 |
| `max_depth` | 5 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 4 |

✅ **Best CV Accuracy**: 83.28%

---

## 💾 8. Model Export

```python
import joblib
joblib.dump(best_model, 'best_random_forest_model.pkl')
```

- Model saved in `.pkl` format for reuse and deployment.

---

## 📊 9. Key Insights & Summary

| Insight | Conclusion |
|--------|------------|
| Gender | Females had much higher survival rates |
| Class | 1st class passengers survived more often |
| Age | Children under 10 showed higher survival rates |
| Alone vs Family | People traveling alone had lower survival rates |
| Alcohol & Fare (feature importance) | Highly ranked by Random Forest & XGBoost |

---

## 🧠 10. Final Reflections

This project showcased the importance of:
- Data visualization in uncovering trends
- Careful feature engineering (e.g., `IsAlone`)
- Model selection & tuning
- Reproducibility via pipeline + model export

It mirrors real-world machine learning workflows and could be adapted to production-level deployment with minor modifications.

---

## 🧰 11. How to Use This Repo

```bash
# 1. Clone repo
$ git clone https://github.com/your-username/titanic-survival-prediction.git

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. Run notebooks step-by-step
```

---

## 📌 12. References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

> If you find this project helpful or insightful, please ⭐️ star the repository! Thanks for reading!
