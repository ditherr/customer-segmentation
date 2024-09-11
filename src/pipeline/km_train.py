import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils import load_object


# K-Means model
def kmeans_model(k, data, random):      
    model = KMeans(n_clusters=k, init='k-means++', random_state=random)
    model.fit(data)
    
    return model


# Find the Best K for the model
def find_best_k(data):  
    silhouette_coefficients = []
    
    for k in range(2,11):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)

    # bestK + 2 --> because i initiallize the range start from 2
    best_k = np.argmax(silhouette_coefficients)+2 
    
    return best_k


class CustomData():
    def __init__(self, gender, age, income, spending):
        self.customerId = 0
        self.gender = gender
        self.age = age
        self.income = income
        self.spending = spending
        
    def to_df(self):
        try:
            feature = {
                "Customer ID": [self.customerId],
                "Gender": [self.gender],
                "Age": [self.age],
                "Annual Income (k$)": [self.income],
                "Spending Score (1-100)": [self.spending],
            }
            
            return pd.DataFrame(feature)
        
        except Exception as e:
            print(e)
        

def predict_pipe(df):
    #load scaler
    scaler_path = os.path.join('data', 'sscaler.pkl')
    model_path = os.path.join('data', 'model_xgb.pkl')
    
    scaler = load_object(scaler_path)
    model = load_object(model_path)
    
    data = df.iloc[:, 1:5]
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    predict_scaler = scaler.transform(data)
    pred = model.predict(predict_scaler)
    
    return pred