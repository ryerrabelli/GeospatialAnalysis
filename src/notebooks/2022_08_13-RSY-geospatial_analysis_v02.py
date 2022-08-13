#!/usr/bin/env python
# coding: utf-8

# Created a notebook so it can be organized. Started Aug 12, 2022

# # Set up

# In[18]:


#@title ## Base imports
import os
import sys
import numpy as np
import scipy
import sklearn
import sklearn.linear_model
import pandas as pd
import plotly.express as px

import warnings
import requests
import urllib.request
import json


# In[4]:


#@title ## Option 1) Mount google drive and import my code

mountpoint_folder_name = "gdrive"  # can be anything, doesn't have to be "drive"
project_path_within_drive = "PythonProjects/GeospatialAnalysis" #@param {type:"string"}
project_path_full = os.path.join("/content/",mountpoint_folder_name,
                        "MyDrive",project_path_within_drive)
try:
    import google.colab.drive
    import os, sys
    # Need to move out of google drive directory if going to remount
    get_ipython().run_line_magic('cd', '')
    # drive.mount documentation can be accessed via: drive.mount?
    #Signature: drive.mount(mountpoint, force_remount=False, timeout_ms=120000, use_metadata_server=False)
    google.colab.drive.mount(os.path.join("/content/",mountpoint_folder_name), force_remount=True)  # mounts to a folder called mountpoint_folder_name

    if project_path_full not in sys.path:
        pass
        #sys.path.insert(0,project_path_full)
    get_ipython().run_line_magic('cd', '{project_path_full}')
    
except ModuleNotFoundError:  # in case not run in Google colab
    import traceback
    traceback.print_exc()


# In[2]:


#@title ## Option 2) Clone project files from GitHub

get_ipython().system('git clone https://github.com/ryerrabelli/GeospatialAnalysis.git')

project_path_full = os.path.join("/content/","GeospatialAnalysis")
sys.path.insert(1,project_path_full)
get_ipython().run_line_magic('cd', 'GeospatialAnalysis')
print(sys.path)


# # Helper functions

# In[20]:


years = np.arange(2015,2019+1)

#@markdown # calc_regression()
def calc_regression(y, x):
    import collections
    regress = scipy.stats.linregress(x, y=y)
    # R = Pearson coefficient
    # p indicates p-value
    # Use ordered dict to retain order
    return collections.OrderedDict({
        "Mean": np.mean(y, axis=0),
        "Sum": np.sum(y, axis=0),
        "Slope": regress.slope, 
        "Intercept": regress.intercept, 
        "R": regress.rvalue, 
        "p": regress.pvalue, 
        })


"""
requests.get('http://172.28.0.2:9000/api/sessions').json() =
[{'id': 'e0a49454-e812-4d99-aa6f-9d7b80a9616c',
  'kernel': {'connections': 1,
   'execution_state': 'busy',
   'id': '6fd9c8b4-6573-4ffa-a7d6-f56803a0092c',
   'last_activity': '2021-12-01T08:12:08.239708Z',
   'name': 'python3'},
  'name': 'ModelInversions.ipynb',
  'notebook': {'name': 'ModelInversions.ipynb',
   'path': 'fileId=1ZAqQEIxR08eODSPEHvKPwbZoioVdV8L9'},
  'path': 'fileId=1ZAqQEIxR08eODSPEHvKPwbZoioVdV8L9',
  'type': 'notebook'}]
"""
notebook_filename = requests.get('http://172.28.0.2:9000/api/sessions').json()[0]['name']

#@markdown # get_path_to_save()
def get_path_to_save(file_prefix="", save_filename:str=None, save_in_subfolder:str=None, extension="png", create_folder_if_necessary=True):
    save_path = ["..","outputs",
                f"{notebook_filename.split('.',1)[0]}",  # use split to remove file extension
                ]
    if save_in_subfolder is not None:
        if isinstance(save_in_subfolder, (list, tuple, set, np.ndarray) ):
            save_path.append(**save_in_subfolder)
        else:  # should be a string then
            save_path.append(save_in_subfolder)
    save_path = os.path.join(*save_path)
    if not os.path.exists(save_path) and create_folder_if_necessary:
        os.makedirs(save_path)
    return os.path.join(save_path, file_prefix+save_filename+"."+extension)

#@markdown # save_df()
def save_df(df, file_name:str, ):
    df.to_excel( get_path_to_save(save_filename=file_name, extension="xlsx") )
    df.to_csv( get_path_to_save(save_filename=file_name, extension="csv") )

#@markdown # save_figure()
def save_figure(fig, file_name:str, animated=False):
    """
    fig is of type plotly.graph_objs._figure.Figure,
    Requires kaleido to be installed
    """
    fig.write_html( get_path_to_save(save_filename=file_name, extension="html") )
    if not animated:
        fig.write_image( get_path_to_save(save_filename=file_name, extension="svg"))
        fig.write_image( get_path_to_save(save_filename=file_name, extension="png") )
        fig.write_image( get_path_to_save(save_filename=file_name, extension="jpg") )


# # Procedures analysis

# ## Load ENT procedures df from csv file
# This is specifically a wide type df so it is one row per procedure with years as different columns.To understand what is meant by long type and wide type dataframes, see https://towardsdatascience.com/visualization-with-plotly-express-comprehensive-guide-eb5ee4b50b57

# In[22]:


df_procedures_orig = pd.read_csv("data/2022_05_05 sums and slopes ent with HCPCS descriptions.csv", 
                           dtype={
                               "HCPCS Code": str,
                               "Total Number of Services": np.int64,
                               **{f"Total Number of Services: {year}": np.int64 for year in range(2015,2019+1)}
                               })  # gets per healthcare code info


# The slope given in the csv file is actually the inverse slope. We need to either recalculate it or invert it. I will just recalculate all the regression values.

# ## Clean df and recalculate regression

# In[23]:


df_procedures_clean = df_procedures_orig.set_index(["HCPCS Code", "HCPCS Description"])

# Remove the "amount" word 
df_procedures_clean.columns = [col.replace("Total Medicare Payment Amount","Total Medicare Payment") for col in df_procedures_clean.columns]
# Drop columns besides the individual year ones. Will recalculate the other ones as a quality assurance check.
df_procedures_clean = df_procedures_clean.drop(columns=[col for col in df_procedures_clean.columns if ("slope" in col.lower() or "pearson" in col.lower() or ":" not in col)] )

# Rename the columns so they can be split  easier. The 20 is the first two digits of the year columns
df_procedures_clean.columns = [col.replace(": ",": : ").replace(": 20","Annual: 20") for col in df_procedures_clean.columns]

# Make Multiindex
df_procedures_clean.columns = pd.MultiIndex.from_tuples([tuple(col.split(": ")) if ":" in col else (col,"","") for col in df_procedures_clean.columns], names=["Category","Stat","Year"])
df_procedures_clean = df_procedures_clean[sorted(df_procedures_clean)]  # rearrange cols alphabetically


categories = df_procedures_clean.columns.levels[0]  #["Total Number of Services", "Total Medicare Payment Amount"]

# Calculate regression and sum and mean from individual year later
df_procedures_recalc = df_procedures_clean.copy()
for category in categories:
    new_df = df_procedures_recalc[(category,"Annual")].apply(calc_regression,axis=1, result_type="expand", args=(years,) )
    df_procedures_recalc[[(category,"Overall",new_col) for new_col in new_df.columns ]]=new_df
    #df_procedures_recalc[(category,"","Slope")]=df_procedures_recalc[(category,"Annual")].apply(calc_regression,axis=1)

# rearrange cols alphabetically, but only by the first two elements of the each column's name tuple
# This allows the order of the newly added columns to remain relative to themselves, but be rearranged relative to the other columns
df_procedures_recalc = df_procedures_recalc[sorted(df_procedures_recalc.columns, key=(lambda x: x[0:2]))]  

#df_procedures = df_procedures.sort_values(by=("Total Number of Services","","Sum"), ascending=False)  # sort rows by volume 
df_procedures_recalc = df_procedures_recalc.sort_values(by=("Total Medicare Payment","Overall","Mean"), ascending=False)  # sort rows by volume 


# In[25]:


with pd.option_context('display.float_format', '{:,.2f}'.format):
    display(df_procedures_recalc)

save_df(df_procedures_recalc, "df_procedures_recalc")


# # County analysis

# ## Load data

# In[4]:


# @title Load spatial coordinates of counties
# Below is necessary for plotting chloropleths. 
with urllib.request.urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[5]:


# @title Load conversion df between FIPS code and county string
fips2county = pd.read_csv("data/fips2county.tsv", sep="\t", comment='#', dtype=str)


# In[6]:


# @title Load our ENT df of all counties, their info, and the Moran's analysis
# The ent CSV file only contains the counties which are analyzable
df_counties_wide_orig = pd.read_csv("data/2022_04_10 ent initial output.csv", dtype={"FIPS": str})


# In[7]:


# Merge with the fips 2 county standard data set
df_counties_wide = pd.merge(left=df_counties_wide_orig, right=fips2county, how="left", left_on='FIPS', right_on='CountyFIPS')
# Insert a county "County, ST" col (i.e. "Monmouth, NJ" or "Champaign, IL") for ease
df_counties_wide.insert(1, "County_St", df_counties_wide["CountyName"].astype(str) + ", " + df_counties_wide["StateAbbr"].astype(str))


# In[16]:


info_simple = ["FIPS", "CountyName","StateAbbr", "% ASC Billing", "Moran I score for ACS billing fraction"]
info_main = ["FIPS", "County",	"StateFIPS", "Total Medicare Payment Amount", "% ASC Procedures", "% ASC Billing",	"CountyFIPS_3",	"CountyName",	"StateName",	"CountyFIPS",	"StateAbbr",	"STATE_COUNTY"]

df_counties_wide_simple=df_counties_wide[info_simple]
df_counties_wide_main=df_counties_wide[info_main]

# Display with all the columns
with pd.option_context('display.max_rows', 3, 'display.max_columns', None): 
    display(df_counties_wide_simple)


# ## Create long df from wide df- i.e. separate out the year columns into different rows

# In[12]:


col_categories = ["Total Number of Services:", "Total Medicare Payment Amount:", "% ASC Procedures:", "% ASC Billing:"]
cols_to_keep = ["FIPS","County_St"]  # columns to keep in every subgroup so you can line up extra info later

# Create list of df's to combine later, each df is from melting of one category of columns
df_counties_longs = []

# Convert each type of category to long format in separate dataframes
for col_category in col_categories:
        df_counties_long = df_counties_wide.melt(id_vars=cols_to_keep, 
                               var_name="Year", 
                               value_vars=[f"{col_category} {year}" for year in range(2015, 2019 +1)], 
                               value_name=f"{col_category} in Year",
                               )
        df_counties_long["Year"] = df_counties_long["Year"].replace({ f"{col_category} {year}":f"{year}" for year in range(2015, 2019 +1)})
        df_counties_longs.append(df_counties_long)

# Merge the separate category dataframes
df_counties_long = df_counties_longs[0]
for ind in range(1,len(df_counties_longs)):
    df_counties_long = pd.merge(left=df_counties_long, right=df_counties_longs[ind], how="outer", on=(cols_to_keep+["Year"]) )

# Merge with the overall wide dataframe to keep those other values
df_counties_long = pd.merge(left=df_counties_long, 
                   right=df_counties_wide.drop([f"{col_category} {year}" for year in range(2015, 2019 +1) for col_category in col_categories], axis=1), 
                   how="left", on=cols_to_keep)


# ## Create summary data by Moran category

# In[15]:


categories = ["Total Number of Services","Total Medicare Payment Amount", "% ASC Procedures", "% ASC Billing" ]
# sorted_moran_values = df_counties_wide["Moran I score for ACS billing fraction"].unique()
sorted_moran_values = ["High-High","Low-Low","Low-High","High-Low","Non Significant"]  # list out specifically so you can get the order you want
sorted_moran_values_all = sorted_moran_values + ["All"]   #[pd.IndexSlice[:]]  # pd.IndexSlice[:]] represents all

value_counts = df_counties_wide["Moran I score for ACS billing fraction"].value_counts()[sorted_moran_values]

df_counties_with_slope = df_counties_wide.copy()
# Calculate regression and sum and mean from individual year later
for category in categories:
    new_df = df_counties_with_slope[ [category + ": " + str(yr) for yr in years] ].apply(calc_regression,axis=1, result_type="expand", args=(years,) )
    df_counties_with_slope[[category+": "+new_col for new_col in new_df.columns ]]=new_df
# To simplify, drop info for specific years unless it was "Mean" and "Slope" categories we just added
for category in categories:
    df_counties_with_slope = df_counties_with_slope.drop(columns=[col for col in df_counties_with_slope.columns if category in col and "Mean" not in col and "Slope" not in col])


df_counties_summary_dict = {}   # create a dict we will concatenate into a df later
# Options: 	[count, mean, std, min, 25%, 50%, 75%, max] assuming default percentiles argument
cols_to_show = ["10%","mean","90%"]
for possible_Moran_value in sorted_moran_values:
    df_counties_summary_dict[possible_Moran_value] = df_counties_with_slope[df_counties_with_slope["Moran I score for ACS billing fraction"]==possible_Moran_value].describe(percentiles=[.1,.25,.5,.75,.9]).loc[cols_to_show]
df_counties_summary_dict["All"] = df_counties_with_slope.describe(percentiles=[.1,.25,.5,.75,.9]).loc[cols_to_show]

df_counties_summary = pd.concat(df_counties_summary_dict.values(), axis=0, keys=df_counties_summary_dict.keys())
for possible_Moran_value in sorted_moran_values:
    df_counties_summary.loc[(possible_Moran_value,cols_to_show[0]), "N"] = value_counts[possible_Moran_value]

# Reorder into the sorted order we set above
df_counties_summary = df_counties_summary.loc[sorted_moran_values_all]


# ## Create a more presentable format
# Select out only the columns you want and rename the columns

# In[27]:


key_cols={
    'Total Number of Services: Mean': 'Number of Services: Annual' ,
    'Total Number of Services: Slope': 'Number of Services: Yearly Change' ,
    'Total Medicare Payment Amount: Mean' : "Medicare Payment: Annual",
    'Total Medicare Payment Amount: Slope' : "Medicare Payment: Yearly Change",
    '% ASC Billing: Mean': '% ASC Billing',
    '% ASC Billing: Slope': '% ASC Billing: Yearly Change',
    "% ASC Procedures: Mean": "% ASC Procedures",
    "% ASC Procedures: Slope": "% ASC Procedures: Yearly Change",
    "Average Age": "Average Age (years)",
    'Percent Male': "% Male",
    'Percent Non-Hispanic White': "% Non-Hispanic White",
    'Percent African American': "% African American",
    'Percent Hispanic': "% Hispanic",
    'Percent Eligible for Medicaid': "% Eligible for Medicaid",
    'pct_poverty': "% Poverty",
    'median_house_income': "Median Household Income",
    'unemployment': "Unemployment Rate",
    'pct_uninsured': "% Uninsured",
    'tabacco': "% Tobacco Use",
    'obesity': "% Obesity",
    #"Asthma": "% with Asthma",
    '2013_Rural_urban_cont_code': "RUCA",
    'pop': "Overall Population",
    'Beneficiaries with Part A and Part B': "Medicare Beneficiaries Population",
    'Population Density': "Overall Population Density",
    'Medicare Population Density': "Medicare Population Density",
}
df_counties_long["Year"].replace({ f"{col_category} {year}":f"{year}" for year in range(2015, 2019 +1)})
df_counties_summary_clean = df_counties_summary[key_cols.keys()]
df_counties_summary_clean = df_counties_summary_clean.rename(columns=key_cols).transpose()

with pd.option_context('display.float_format', '{:,.2f}'.format):
    display( df_counties_summary_clean )

save_df(df_counties_summary_clean, "df_counties_summary_clean")


# In[ ]:




