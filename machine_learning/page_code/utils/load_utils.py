import joblib

def model_load(model_path = None,method = "joblib"):
    """
        model load
    """
    model = None
    if method == "joblib":
        model = joblib.load(filename=model_path)
    else:
        pass
    return model

if __name__ == "__main__":
    model = model_load("./../model/knn.test")
    y_pred = model.predict(X=[[2,2,2,3]])

    print(y_pred)