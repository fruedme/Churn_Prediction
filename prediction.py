import joblib


def predict(data):
    clf = joblib.load("xgc_model.sav")
    return clf.predict(data)
