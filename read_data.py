import numpy as np
import pandas as pd

'''
This file contains a bunch of simple functions that use pd.dataframe methods to 
print out different features of our dataset.
They each take in the dataframe `df` that is created in the example_main function in
part2_house_value_regression.py

'''

def first_and_last_five_rows(df):
    print("Here's the first and last five rows of the dataset:")
    print(df.head())
    print("\n")
    print(df.tail())
    print("\n")

def summary_statistics(df):
    print("Here are some summary statistics:")
    print(df.describe())
    print("\n")

def dataset_datatypes(df):
    print("Here are the datatypes for each column")
    print(df.dtypes)
    print("\n")
    print("Let's take a look at that ocean_proximity column. (the only categorical one)")
    print(df["ocean_proximity"].value_counts())
    print("\n")

def missing_values(df):
    numerical_columns = df.columns[df.dtypes != 'object']
    categorical_columns = df.columns[df.dtypes == 'object']
    print("Finding the percentage of missing values for numerical columns")
    print(df[numerical_columns].isnull().sum().sort_values(ascending = False)/len(df))
    print("\n")
    print("And for the categorical column")
    print(df[categorical_columns].isnull().sum().sort_values(ascending = False)/len(df))
    

    

