#!/usr/bin/env python
# coding: utf-8

# In[41]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:





# In[42]:


X = pd.read_pickle("x1.pkl")
y = pd.read_pickle("y1.pkl")


# In[44]:


X.to_pickle("x11.pkl", protocol = 4)
y.to_pickle("y11.pkl", protocol = 4)


# In[36]:


#X.columns


# In[37]:


#x["epoch"].unique()


# In[ ]:





# In[38]:


def period_to_numeric(a):
    if a < 1919:
        return 1
    if a > 1918 and a < 1946:
        return 1
    if a > 1945 and a < 1961:
        return 3
    if a > 1960 and a < 1971:
        return 4
    if a > 1970 and a < 1981:
        return 5
    if a > 1980 and a < 1991:
        return 6
    if a > 1990 and a < 1996:
        return 7
    if a > 1995 and a < 2001:
        return 8
    if a > 2000 and a < 2006:
        return 9
    else:
        return 9


# In[6]:


from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)


# In[9]:


st.write("""
# Lisbon Energy consumption prediction app

This app predicts the **energy consumption** of your house/building in Lisbon!
""")
st.write("---")


# In[46]:


X.rot.min()


# In[47]:


X.rot.max()


# In[14]:


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Period = st.sidebar.slider('Ano de Construção', 1918, 2021, 1960)
    f_area = st.sidebar.slider('Área útil', 50, 800, 100)
    prop = st.sidebar.slider('Proporção', 1.00, 5.00, 2.00, 0.01)
    N_floors = st.sidebar.slider('Número de pisos', round(X.nfloor.min()), round(X.nfloor.max()), round(X.nfloor.mean()))
    rot = st.sidebar.slider('Orientação radianos em relação a Este', round(X.rot.min(), 2), round(X.rot.max(), 2), 0.00)
    wwr = st.sidebar.slider('Rácio de envidraçado', round(X.wwr.min(), 2), round(X.wwr.max(), 2), 0.3)
    data = {"Número de pisos": N_floors,
            "Proporção rectangular do apartamento": prop,
            "Orientação radianos em relação a Este": rot,
            "Área útil": f_area,
            "Rácio de envidraçado": wwr,
            "Ano de Construção":period_to_numeric(Period),
            "Solução Construtiva - Parede": 1,
            "Solução Construtiva - Cobertura": 1,
            "Solução Construtiva - Pavimento": 1}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# In[15]:


df


# In[16]:


prediction = model.predict(df)


# In[18]:


st.header('Prediction of Energy Consumption in kWh/m2 and Euros')
st.write(round(prediction[0], 2))
st.write(round(prediction[0]*0.15252*df["Área útil"][0], 2), "€")
st.write('---')


# In[33]:


st.header('Specify Desired upgrade')
st.checkbox("Tipo de renovação", ["Paredes", "Janelas", "Águas quentes e sanitárias"])


# In[ ]:




