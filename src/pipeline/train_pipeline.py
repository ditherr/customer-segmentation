import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objs as go



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


def kmeans_model(k, data):
    model = KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    
    return model




