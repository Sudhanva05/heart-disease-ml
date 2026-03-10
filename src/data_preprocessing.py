import pandas as pd
import numpy as np


def load_and_clean_data(path):

    df = pd.read_csv(path, names=[
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"
    ])

    # Replace ? with NaN
    df = df.replace("?", np.nan)

    # Convert to numeric
    df = df.apply(pd.to_numeric)

    # Binary target
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Fill missing values
    df = df.fillna(df.median())

    return df