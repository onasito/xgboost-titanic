import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb


def load_data(filepath):
    df = pd.read_csv(filepath)

    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Cabin'] = df['Cabin'].notnull().astype(int)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df


def main():
    df = load_data('train.csv')

    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    print(f"Validation accuracy: {accuracy_score(y_val, predictions):.4f}")

    model.save_model('model.json')
    print("Model saved to model.json")


if __name__ == '__main__':
    main()

