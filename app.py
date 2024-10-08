# Build App to Predict Cluster and Give the Information for each Clusters

import pandas as pd
import numpy as np
import streamlit as st
import time
from src.pipeline.km_train import CustomData
from src.pipeline.km_train import predict_pipe
import plotly.express as px
import plotly.graph_objs as go
from src.analysis import cluster2d, cluster3d, load_model, convert_df, show_gender, clustered

## Dataset
dataset = pd.read_csv('data/Mall_Customers.csv')
sample = pd.read_csv('data/Mall_Customers.csv')
selected_feature = ["Annual Income (k$)", "Spending Score (1-100)"]
selected_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = dataset[selected_feature]
X2 = dataset[selected_features]


def main():
    st.set_page_config(
        page_title="Customer Segmentation",
        page_icon="🏬",
        layout="centered"
    )
    
    st.title("Customer Segmentation")
    st.markdown("""This dataset contains Customer behavior based on their annual income and expenditure scores in a Mall""")
    
    st.subheader('📋Existing Dataset')
    dataset.columns = ['Customer ID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    with st.expander("👀 Show Datasets"):
        st.page_link("https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python", label="Kaggle - Mall Customer Segmentation", icon="🗃️")
        st.write(dataset)
    
    st.subheader('🕹️Play with Clustering')
    st.markdown("""Play with clustering, how the cluster can affect the data. :blue-background[**Let's try and see**].\
        Show in two style, Cluster 2D and Cluster 3D, you can choose both.""")
    with st.expander("🗺️ Do clustering"):
        cl, ch = st.columns(2)
        with cl:
            cluster = st.slider("Choose Cluster", 1, 10)
            random = st.slider("Random State (try to see the different)", 2, 12, step=2)
        with ch:
            st.write("**Choose the Cluster style**")
            cl2d = st.checkbox("Cluster 2D")
            cl3d = st.checkbox("Cluster 3D")
            
        if cl2d:
            data1d, model = load_model(cluster, X, random)
            cluster2d(data1d, model)
        if cl3d:
            data2d, model = load_model(cluster, X2, random)
            cluster3d(data2d, model)
    
    st.subheader('🕵️Try to Predict')
    st.markdown("""The prediction use the dataset above, you can use your dataset but you need to :blue-background[**have same features or columns like ours**].\
        You can do the prediction by input a single data or using our sample here:""")


    csv = convert_df(sample)

    st.download_button(
        label="Download sample",
        data=csv,
        file_name="sample.csv",
        mime="text/csv",
    )
    with st.expander("🕵️ Predict please...."):
        st.info("ℹ️ - **prediction will use our model (*KMeans with K=5*)**")
        pred = st.radio("Pick your options...", options=["Single", "Datasets"])
        
        if pred == 'Single':
            with st.form('Predict Cluster', clear_on_submit=True):
                p1, p2 = st.columns(2)
                with p1:
                    gender = st.selectbox("Gender",['Male', 'Female'], index=None, placeholder='Gender...')
                    age = st.number_input("Age", min_value=17, max_value=70, step=1, placeholder="Age....")
                with p2:
                    income = st.number_input("Anual Income (k$)", min_value=10, max_value=150, step=1, placeholder="Annual Income....")
                    spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, step=1, placeholder="Your Age....")
                
                submitted = st.form_submit_button('Predict')
                
                if submitted:
                    if not gender or not age or not income or not spending: 
                        st.warning('Please completed the input field !!')
                    else:
                        data = CustomData(gender, age, income, spending)
                        df = data.to_df()
                        
                        # predict the result
                        result = predict_pipe(df)
                        if result == 0:
                            st.success(f'Based on your input, You are clustered into **Cluster {result}** and joined by people who **earn high but spend less**.', icon="🔎")
                        elif result == 1:
                            st.success(f'Based on your input, You are clustered into **Cluster {result}** and joined by people who **more average in terms of earn and spend**.', icon="🔎")
                        elif result == 2:
                            st.success(f'Based on your input, You are clustered into **Cluster {result}** and joined by people who **earn high also spend more**.', icon="🔎")
                        elif result == 3:
                            st.success(f'Based on your input, You are clustered into **Cluster {result}** and joined by people who **earn less but spend more**.', icon="🔎")
                        else:
                            st.success(f'Based on your input, You are clustered into **Cluster {result}** and joined by people who **earn less also spend less**.', icon="🔎")
            

        else:
            uploaded_file = st.file_uploader("Choose a CSV file")
            
            if uploaded_file is not None:
                data_sample = pd.read_csv(uploaded_file)
                result = predict_pipe(data_sample)
                data_sample['Cluster'] = result
                
                progress_text = "Your Data is being Clustered and Analyzed. Please wait."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()
                
                # Show success message
                st.success("🎉 **Congrats! Your data has been successfully clustered**")
                con = st.container()
                con.write(data_sample)
                note = '''
                **Note**:
                - **`cluster [0]`** ➡ earn high but spend less.
                - **`cluster [1`]** ➡ more average in terms of earning and spending.
                - **`cluster [2]`** ➡ earn high, spend more.
                - **`cluster [3]`** ➡ earn less but spend more.
                - **`cluster [4]`** ➡ earn less, spend less.
                '''
                
                con.markdown(note)
                
                
                show_gender(data_sample)
                clustered(data_sample)
            else:
                st.warning('Please upload the datasets')
                
            

            
            
    
if __name__ == '__main__':
    main()