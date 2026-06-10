import pandas as pd
import xgboost as xgb


def load_model(filepath='model.json'):
    model = xgb.XGBClassifier()
    model.load_model(filepath)
    return model


def prepare_passenger(pclass, sex, age, sibsp, parch, fare, cabin, embarked):
    return pd.DataFrame([{
        'Pclass': pclass,
        'Sex': 0 if sex == 'male' else 1,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Cabin': 1 if cabin else 0,
        'Embarked': {'S': 0, 'C': 1, 'Q': 2}[embarked]
    }])


def predict(model, passenger_df):
    prediction = model.predict(passenger_df)[0]
    probability = model.predict_proba(passenger_df)[0][1]
    return prediction, probability


def main():
    model = load_model()

    passenger = prepare_passenger(
        pclass=3,
        sex='male',
        age=22,
        sibsp=1,
        parch=0,
        fare=7.25,
        cabin=False,
        embarked='S'
    )

    prediction, probability = predict(model, passenger)
    survived = "Survived" if prediction == 1 else "Did not survive"
    print(f"Prediction: {survived} (probability: {probability:.2%})")


if __name__ == '__main__':
    main()
