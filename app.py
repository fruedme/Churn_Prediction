import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Customer Churn Prediction')
st.markdown('Use customers features to predict whether this customer is going to churn or not')

st.header("Customer Profile")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    Tenure = st.slider('Tenure', 0, 31, 1)
    Complain = st.slider('Complain', 0, 1, 1)

with col2:
    st.text("Pepal characteristics")
    Cashback = st.slider('CashbackAmount', 110, 324, 1)
    SatisfactionScore = st.slider('SatisfactionScore', 1, 5, 1)

st.text('')
if st.button("Customer churn prediction"):
    result = predict(
        np.array([[Tenure, Complain, Cashback, SatisfactionScore]]))
    st.text(result[0])


st.text('')
st.text('')
st.markdown(
    '`Create by` [santiviquez](https://twitter.com/santiviquez) | \
         `Code:` [GitHub](https://github.com/santiviquez/iris-streamlit)')