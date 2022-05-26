# Importing libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import pickle

df_train = df_data_source('https://raw.githubusercontent.com/sets018/Ocelot/main/data_extraction/df_posts_housing_clean_final.csv','url',0.9,0.1)
sectors = borough_classifier(df_train)
sectors.get_sectors()

encoder = oh_encoder(df_train.data_source)
df_encoded_train = df_data_source(encoder.encode(),'pass',0.9,0.1)

grad_boost = Predictor('gradient_boosting',df_encoded_train,0.7,0.3)
params = {
    "model__regressor__n_estimators": [300],
    "model__regressor__max_depth": [5],
    "model__regressor__learning_rate": [0.1],
}
tuner = tuning(grad_boost,"r2",params)
grad_boost.fit_print_scr()

with open("trained_grad_boost_model.bin", 'wb') as f_out:
    pickle.dump(grad_boost.reg, f_out) # write final_model in .bin file
    f_out.close()  # close the file 
