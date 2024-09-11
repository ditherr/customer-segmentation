# Helper function to analysis
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.pipeline.km_train import kmeans_model
from src.utils import load_object

dataset = pd.read_csv('data/Mall_Customers.csv')
coloring = ["red", "green", "blue", "purple", "orange", "magenta", "cyan", "yellow", "brown", "indigo"]

## Transform as a string for coloring purpose
def transform_cluster(data):
    data = data.sort_values(by='Cluster', ascending=True)
    data['Cluster'] = data['Cluster'].apply(lambda x: x+1)
    data['Cluster'] = data['Cluster'].astype(str)
    
    return data


## Load model
def load_model(cluster, model, data):
    if model == 1:
        model = kmeans_model(cluster, data)
        data_clustered = dataset.copy()
        data_clustered['Cluster'] = model.labels_
        data = transform_cluster(data_clustered)
        return data, model
    elif model == 2:
        model = kmeans_model(cluster, data)
        data_clustered = dataset.copy()
        data_clustered['Cluster'] = model.labels_
        data = transform_cluster(data_clustered)
        return data, model


## Draw a 2D Cluster
def cluster2d(data, model):  
    fig = px.scatter(data,
                 x="Annual Income (k$)",
                 y="Spending Score (1-100)",
                 color="Cluster",
                 color_discrete_sequence=coloring)
    
    # Add cluster centers to the plot
    fig.add_trace(go.Scatter(
        x=model.cluster_centers_[:, 0],
        y=model.cluster_centers_[:, 1],
        mode='markers',
        marker=dict(color='black', size=15, symbol='x'),
        name='Cluster Centers'
    ))

    fig.update_traces(marker_size=10)
    fig.update_layout(title='Clustering in 2D', plot_bgcolor='white', showlegend=True)
        
    st.plotly_chart(fig)
    

## Draw a 3D Cluster
def cluster3d(data, model):
    fig = px.scatter_3d(data,
                x='Age',
                y='Annual Income (k$)',
                z='Spending Score (1-100)',
                color="Cluster",
                color_discrete_sequence=coloring,
                size_max=10)
    
    # Add cluster centers to the 3D plot
    fig.add_trace(go.Scatter3d(
        x=model.cluster_centers_[:, 0],  # Age
        y=model.cluster_centers_[:, 1],  # Annual Income
        z=model.cluster_centers_[:, 2],  # Spending Score
        mode='markers',
        marker=dict(color='black', size=6, symbol='x'),
        name='Cluster Centers'
    ))
    
    fig.update_layout(title='Clustering in 3D', plot_bgcolor='white')

    
    st.plotly_chart(fig)
    
def clustered(data):
    data = transform_cluster(data)

    fig = px.scatter_3d(data,
                x='Age',
                y='Annual Income (k$)',
                z='Spending Score (1-100)',
                color="Cluster",
                color_discrete_sequence=coloring,
                size_max=10)
    
    fig.update_layout(title='Clustering in 3D', plot_bgcolor='white')

    st.plotly_chart(fig)
    
    
    
# Downloading PDF
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


# Load a file
def upload_file(file):
    if file is not None:
        dataframe = pd.read_csv(file)
    
        return dataframe
    else:
        st.warning('Please upload the datasets')
        
        
# Kmeans label predict
def show_gender(data):
    color_list = ['#EB455F', '#BAD7E9']
    count_gender = data['Gender'].value_counts()
    df = pd.DataFrame({'Gender': count_gender.index, 'Total': count_gender.values})

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])

    bar_trace = go.Bar(x=df.Gender, y=df.Total, marker=dict(color=color_list), text=df.Total, textposition='auto')
    pie_trace = go.Pie(labels=df.Gender, values=df.Total, marker=dict(colors=color_list), textinfo='percent+label')

    fig.add_trace(bar_trace, row=1, col=1)
    fig.add_trace(pie_trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        title_text="Gender Distribution in Dataset",  # Title for the entire figure
        yaxis=dict(title="Count", gridcolor='lightgray'),
        showlegend=False,  # Disable the shared legend
        plot_bgcolor='white',  # Set plot background to white
        width=720
    )

    # Display the figure
    st.plotly_chart(fig)