import joblib
import pandas as pd
import pickle

def wrangle(test_file):
    df = pd.read_csv(test_file, index_col='Id')
    
    # Create list of columns with less than 100 values
    cols_to_drop = ['Alley', 'PoolQC', 'MiscFeature', 'GarageCars']

    # Drop columns wit high amount of missing values
    df.drop(columns=cols_to_drop, inplace=True)
    
    return df

def make_predictions(test_path):
    model = joblib.load("my_GBR_model.pkl")
    
    test_data = wrangle(test_path)
    
    prediction = model.predict(test_data)
    
    prediction= pd.Series(prediction, index=test_data.index, name="SalePrice" )
    return prediction