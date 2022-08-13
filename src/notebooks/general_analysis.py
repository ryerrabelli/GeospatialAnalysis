#!/usr/bin/env python
# coding: utf-8

# In[3]:


#@title ## Base imports
import os
import sys
import numpy as np
import scipy
import pandas as pd
import geopandas
import plotly.express as px

import sklearn
import sklearn.linear_model
import statsmodels.api as sm


# ## Load ENT county df (specifically wide-type df) from csv file
# To understand what is meant by long type and wife type dataframes, see https://towardsdatascience.com/visualization-with-plotly-express-comprehensive-guide-eb5ee4b50b57

# In[4]:


df_bill_orig = pd.read_csv("data/2022_05_05 sums and slopes ent with HCPCS descriptions.csv", 
                           dtype={
                               "HCPCS Code": str,
                               "Total Number of Services": np.int64,
                               **{f"Total Number of Services: {year}": np.int64 for year in range(2015,2019+1)}
                               })  # gets per healthcare code info


# In[5]:


df_bill_orig.head(2)


# In[12]:


df_bill_wide = df_bill_orig.set_index(["HCPCS Code", "HCPCS Description"])
# Rename the columns so they can be split  easier. The 20 is the first two digits of the year columns
df_bill_wide.columns = [col.replace(": ",": : ").replace(": 20","Annual: 20") for col in df_bill_wide.columns]
# Multiindex
df_bill_wide.columns = pd.MultiIndex.from_tuples([tuple(col.split(": ")) if ":" in col else (col,"","Sum") for col in df_bill_wide.columns], names=["Category","Stat","Year"])
df_bill_wide = df_bill_wide[sorted(df_bill_wide)]  # rearrange cols alphabetically
df_bill_wide = df_bill_wide.sort_values(by=("Total Number of Services","","Sum"), ascending=False)  # sort rows by volume 
categories = df_bill_wide.columns.levels[0]  #["Total Number of Services", "Total Medicare Payment Amount"]


# The slope given in the csv file is actually the inverse slope. We need to either recalculate it or

# In[63]:


def calc_slope(y, x, invert=False ):
    if invert: 
        # the inverse linear regression does not necessarily have 1/slope of the regular linear regression
        # the "2022_05_05 sums and slopes ent with HCPCS descriptions.csv" contains the inverse linear regression
        temp = x
        x = y 
        y = temp
    regress = scipy.stats.linregress(x, y=y)  # x=np.arange(2015,2019+1)
    return {"Slope": regress.slope, "Pearson Coef": regress.rvalue, "P": regress.pvalue}
    #return {"Slope2": regress.slope, "Pearson Coef2": regress.rvalue, "P": regress.pvalue}
    #return {"Slope": regress.slope, "Pearson Coef": regress.rvalue, "Intercept": regress.intercept, "P": regress.pvalue}


# In[64]:


df_bill_wide_stats = df_bill_wide.copy()
for category in categories:
    new_df = df_bill_wide[(category,"Annual")].apply(calc_slope,axis=1, result_type="expand", args=(np.arange(2015,2019+1),) )
    df_bill_wide_stats[[(category,"",new_col) for new_col in new_df.columns ]]=new_df
    #df_bill_wide[(category,"","Slope")]=df_bill_wide[(category,"Annual")].apply(calc_slope,axis=1)
df_bill_wide_stats = df_bill_wide_stats[sorted(df_bill_wide_stats.columns)]  # rearrange cols alphabetically


# In[65]:


df_bill_wide_stats.head(2)


# In[54]:


1/-222163.937	


# In[45]:


df_bill_wide


# In[ ]:




