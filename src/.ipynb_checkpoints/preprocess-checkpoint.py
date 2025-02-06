import pandas as pd

def preprocess_data(df):
    """ Perform data cleaning and feature engineering. """
    df = df.copy()

    # Fill missing Age values with median
    df = df.assign(Age=df["Age"].fillna(df["Age"].median()))

    # Fill missing Embarked values with mode
    df = df.assign(Embarked=df["Embarked"].fillna(df["Embarked"].mode()[0]))

    # Drop Cabin column (too many missing values)
    df = df.drop(columns=["Cabin"], errors="ignore")

    # Convert categorical variables to numeric
    df = df.assign(Sex=df["Sex"].map({"male": 0, "female": 1}))
    df = df.assign(Embarked=df["Embarked"].map({"S": 0, "C": 1, "Q": 2}))

    # Feature Engineering
    df = df.assign(FamilySize=df["SibSp"] + df["Parch"] + 1)
    df = df.assign(IsAlone=(df["FamilySize"] == 1).astype(int))

    # Extract and map Title
    df = df.assign(Title=df["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False))
    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
        "Dr": 4, "Rev": 4, "Col": 4, "Major": 4,
        "Mlle": 1, "Mme": 2, "Don": 4, "Dona": 4,
        "Lady": 4, "Countess": 4, "Jonkheer": 4, "Sir": 4, "Capt": 4
    }
    df = df.assign(Title=df["Title"].map(title_mapping).fillna(4))

    # Drop unnecessary columns
    df = df.drop(columns=["Name", "Ticket"], errors="ignore")

    return df