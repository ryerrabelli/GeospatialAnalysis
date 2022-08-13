#!/usr/bin/env python
# coding: utf-8

# # Pip install the non-common modules

# In[ ]:


get_ipython().system('pip install -U kaleido   # kaleido required for saving the plotly figures into static images')


# In[ ]:


get_ipython().system('pip install geopandas --quiet')
get_ipython().system('pip install geoplot --quiet')


# In[ ]:


get_ipython().system('pip install watermark')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-d -m -v -p numpy,matplotlib,sklearn,pandas')


# # Set up

# In[ ]:


#@title ## Base imports
import os
import sys
import numpy as np
import scipy
import sklearn
import sklearn.linear_model
import pandas as pd
import plotly.express as px


# In[ ]:


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


# In[ ]:


#@title ## Option 2) Clone project files from Github

get_ipython().system('git clone https://github.com/ryerrabelli/GeospatialAnalysis.git')

project_path_full = os.path.join("/content/","GeospatialAnalysis")
sys.path.insert(1,project_path_full)
get_ipython().run_line_magic('cd', 'GeospatialAnalysis')
print(sys.path)


# ## Set up a folder for saving images

# In[ ]:


image_folder_path = "outputs"
if not os.path.exists(image_folder_path):
    os.mkdir(image_folder_path)

def save_figure(fig, file_name:str, animated=False):
    """
    fig is of type plotly.graph_objs._figure.Figure,
    Requires kaleido to be installed
    """
    fig.write_html(os.path.join(image_folder_path, file_name+".html"))
    if not animated:
        fig.write_image(os.path.join(image_folder_path, file_name+".svg"))
        fig.write_image(os.path.join(image_folder_path, file_name+".png"))
        fig.write_image(os.path.join(image_folder_path, file_name+".jpg"))


# # Load data

# ## Load geojson data

# In[ ]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[ ]:


for k,v in counties.items():
    print(k,": ",v)


# ## Test

# In[ ]:


dfA = pd.read_csv("data/2022_04_10 ent initial output.csv", dtype={"FIPS": str})  # gets per county info
dfB = pd.read_csv("data/2022_05_05 sums and slopes ent.csv", dtype={"HCPCS Code": str})  # gets per healthcare code info


# In[ ]:


display(dfA.info())
display(dfB.info())


# In[ ]:


dfB["HCPCS Code"]


# In[ ]:


for ind in range(max(len(dfA.columns),len(dfB.columns))):
    dfAstr = str(dfA.columns[ind])
    print(dfAstr, " "*(60-len(dfAstr)),
          (dfB.columns[ind] if ind < len(dfB.columns) else "x" ))


# ## Load ENT county df (specifically wide-type df) from csv file
# To understand what is meant by long type and wife type dataframes, see https://towardsdatascience.com/visualization-with-plotly-express-comprehensive-guide-eb5ee4b50b57

# In[ ]:


df_bill_orig = pd.read_csv("data/2022_05_05 sums and slopes ent with HCPCS descriptions.csv", 
                           dtype={
                               "HCPCS Code": str,
                               "Total Number of Services": np.int64,
                               **{f"Total Number of Services: {year}": np.int64 for year in range(2015,2019+1)}
                               })  # gets per healthcare code info


# In[ ]:


df_bill_orig.head(2)


# In[ ]:


df_bill_wide = df_bill_orig.set_index(["HCPCS Code", "HCPCS Description"])
# Rename the columns so they can be split  easier. The 20 is the first two digits of the year columns
df_bill_wide.columns = [col.replace(": ",": : ").replace(": 20","Annual: 20") for col in df_bill_wide.columns]
# Multiindex
df_bill_wide.columns = pd.MultiIndex.from_tuples([tuple(col.split(": ")) if ":" in col else (col,"","Sum") for col in df_bill_wide.columns], names=["Category","Stat","Year"])
df_bill_wide = df_bill_wide[sorted(df_bill_wide)]  # rearrange cols alphabetically
df_bill_wide = df_bill_wide.sort_values(by=("Total Number of Services","","Sum"), ascending=False)  # sort rows by volume 
categories = df_bill_wide.columns.levels[0]  #["Total Number of Services", "Total Medicare Payment Amount"]


# The slope given in the csv file is actually the inverse slope. We need to either recalculate it or

# In[ ]:


def calc_slope(y, x):
    regress = scipy.stats.linregress(x, y=y)
    return {"Slope": regress.slope, "Pearson Coef": regress.rvalue, "Intercept": regress.intercept, "P": regress.pvalue}


# In[ ]:


df_bill_wide.columns


# In[ ]:


for category in categories:
    new_df = df_bill_wide[(category,"Annual")].apply(calc_slope,axis=1, result_type="expand", args=(np.arange(2015,2019+1),) )
    df_bill_wide[[(category,"",new_col) for new_col in new_df.columns ]]=new_df
    #df_bill_wide[(category,"","Slope")]=df_bill_wide[(category,"Annual")].apply(calc_slope,axis=1)
df_bill_wide = df_bill_wide[sorted(df_bill_wide.columns)]  # rearrange cols alphabetically


# ## Nick suppl table

# In[ ]:


df_bill_wide.shape[1]


# In[ ]:


df_bill_wide.loc[["14060",
"69930",
"31267",
"30520",
"30140",
"15260",
"14301",
"14040",
"31575",
"31276",
"14041",
"60500",
"31231",
"31579",
"69436",
"31256",
"31525",
"11042",
"31237"]]


# In[ ]:


df_bill_wide.loc[["31575","31237"]]


# In[ ]:


A = sklearn.linear_model.LinearRegression()
B = A.fit(np.arange(2015,2019+1).reshape(-1,1), df_bill_wide.loc["14060"][("Total Number of Services","Annual")].values.reshape(-1,1) )


# In[ ]:


B.coef_, B.intercept_


# In[ ]:


import statsmodels.api as sm


# In[ ]:


df_bill_wide[(category,"Annual")].apply(calc_slope,axis=1, result_type="expand", args=(np.arange(2015,2019+1),) )


# In[ ]:


X = sm.add_constant(np.arange(2015,2019+1)) # adding a constant
Y = df_bill_wide.loc["14060"][("Total Number of Services","Annual")].values.reshape(-1,1)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()


# In[ ]:


print_model


# ## Data

# In[ ]:


with pd.option_context('display.float_format', '{:,.2f}'.format):
    display(df_bill_wide)


# In[ ]:


print("Total amount of services")
A1 = np.sum(df_bill_wide[("Total Number of Services","Total")])
print("{:,}".format(A1))
print("{:,}".format(A1/5))
print("Total medical payment")
A2 = np.sum(df_bill_wide[("Total Medicare Payment Amount","Total")].astype(np.int64) )
print("{:,}".format(A2))
print("{:,}".format(A2/5))


# In[ ]:


pd.columns = pd.MultiIndex.from_tuples([tuple(col.split(":")) if ":" in col else (col,"") for col in df_bill_wide.columns])


# In[ ]:


pd.MultiIndex.from_tuples([("Gasoline", "Toyoto"), 
                                  ("Gasoline", "Ford"), 
                                  ("Electric", "Tesla"),
                                  ("Electric", "Nio")])


# In[ ]:


fips2county = pd.read_csv("data/fips2county.tsv", sep="\t", comment='#', dtype=str)


# In[ ]:


# The ent CSV file only contains the counties which are analyzable
df_loc_orig = pd.read_csv("data/2022_04_10 ent initial output.csv", dtype={"FIPS": str})
# Merge with the fips 2 county standard data set
df_loc_wide = pd.merge(left=df_loc_orig, right=fips2county, how="left", left_on='FIPS', right_on='CountyFIPS')
# Insert a county "County, ST" col (i.e. "Freehold, NJ" or "Chicago, IL") for ease
df_loc_wide.insert(1, "County_St", df_loc_wide["CountyName"].astype(str) + ", " + df_loc_wide["StateAbbr"].astype(str))
# Display with all the columns
with pd.option_context('display.max_rows', 3, 'display.max_columns', None): 
    display(df_loc_wide)
    pass

loc_simple = ["FIPS", "CountyName","StateAbbr", "% ASC Billing", "Moran I score for ACS billing fraction"]
df_loc_wide_simple=df_loc_wide[loc_simple]

loc_main = ["FIPS", "County",	"StateFIPS", "Total Medicare Payment Amount", "% ASC Procedures", "% ASC Billing",	"CountyFIPS_3",	"CountyName",	"StateName",	"CountyFIPS",	"StateAbbr",	"STATE_COUNTY"]
#a=pd.merge(right=df_loc_orig, left=fips2county, how="outer", right_on='FIPS', left_on='CountyFIPS')
#a=a.loc[:,loc_main]
#df_loc_orig2=df_loc_orig.loc[:,["FIPS","pop","Moran I score for ACS billing fraction","County"]]


# ## Convert wide df to long df - i.e. separate out the year columns into different rows

# In[ ]:


col_categories = ["Total Number of Services:", "Total Medicare Payment Amount:", "% ASC Procedures:", "% ASC Billing:"]
cols_to_keep = ["FIPS","County_St"]  # columns to keep in every subgroup so you can line up extra info later

# Create list of df's to combine later, each df is from melting of one category of columns
df_loc_longs = []

# Convert each type of category to long format in separate dataframes
for col_category in col_categories:
        df_loc_long = df_loc_wide.melt(id_vars=cols_to_keep, 
                               var_name="Year", 
                               value_vars=[f"{col_category} {year}" for year in range(2015, 2019 +1)], 
                               value_name=f"{col_category} in Year",
                               )
        df_loc_long["Year"] = df_loc_long["Year"].replace({ f"{col_category} {year}":f"{year}" for year in range(2015, 2019 +1)})
        df_loc_longs.append(df_loc_long)

# Merge the separate category dataframes
df_loc_long = df_loc_longs[0]
for ind in range(1,len(df_loc_longs)):
    df_loc_long = pd.merge(left=df_loc_long, right=df_loc_longs[ind], how="outer", on=(cols_to_keep+["Year"]) )

# Merge with the overall wide dataframe to keep those other values
df_loc_long = pd.merge(left=df_loc_long, 
                   right=df_loc_wide.drop([f"{col_category} {year}" for year in range(2015, 2019 +1) for col_category in col_categories], axis=1), 
                   how="left", on=cols_to_keep)

display(df_loc_long)


# # Analysis

# In[ ]:




fig = px.choropleth(df_loc_wide, geojson=counties, locations='FIPS', 
                    color='% ASC Procedures',
                    color_continuous_scale="Viridis",
                    #range_color=(0, 12),
                    scope="usa",
                    #facet_col="Moran I score for ACS billing fraction",
                    labels={
                        "2013_Rural_urban_cont_code":"2013-RUCA",
                        "pop":"Pop.",
                        "Average Age":"Mean Age",
                        "Percent Male":"% M",
                        "tot_ratio":"Tot. Ratio",
                        },
                    hover_name="County_St",
                    hover_data={
                        '% ASC Procedures':   ":.0f",
                        "FIPS":True, 
                        "pop":  ":.1f", 
                        "2013_Rural_urban_cont_code":True,
                        "Average Age":  ":.1f", 
                        "Percent Male": ":.1f",
                        },
                    )

fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell",
        align="auto"
    )
)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

# Define layout specificities
fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    title={
        'text': f"% ASC Procedures",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
#fig.show()


save_figure(fig,"choropleth-total")


# In[ ]:


fig = px.choropleth(df_loc_long, geojson=counties, locations='FIPS', 
                    color='% ASC Procedures: in Year',
                    color_continuous_scale="Viridis",
                    #range_color=(0, 12),
                    scope="usa",
                    #facet_col="Moran I score for ACS billing fraction",
                    labels={
                        "2013_Rural_urban_cont_code":"2013-RUCA",
                        "pop":"Pop.",
                        "Average Age":"Mean Age",
                        "Percent Male":"% M",
                        "tot_ratio":"Tot. Ratio",
                        },
                    hover_name="County_St",
                    hover_data={
                        "FIPS":True, 
                        "pop":  ":.1f", 
                        "2013_Rural_urban_cont_code":True,
                        "Average Age":  ":.1f", 
                        "Percent Male": ":.1f",
                        },
                    animation_frame="Year",
                    )

fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell",
        align="auto"
    )
)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

# Define layout specificities
fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    title={
        'text': f"% ASC Procedures by Year",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


save_figure(fig,"choropleth-all", animated=True)


# # Moran's

# In[ ]:


get_ipython().system('pip install rioxarray --quiet')


# In[ ]:


get_ipython().system('pip install pysal --quiet')


# In[ ]:


import geopandas
import rioxarray                 # Surface data manipulation
import xarray                    # Surface data manipulation
from pysal.explore import esda   # Exploratory Spatial analytics
from pysal.lib import weights    # Spatial weights
import contextily                # Background tiles


# In[ ]:


df_loc_wide_simple


# In[ ]:


countiesgeo = geopandas.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")


# In[ ]:


countiesgeo.head()


# In[ ]:


countiesgeo.info()


# In[ ]:


countiesnick = geopandas.read_file("examples/npeterman/ent redo3anova.geojson")


# In[ ]:


countiesnick[["FIPS","LISA_CL"]]


# In[ ]:


# Generate W from the GeoDataFrame
w = weights.distance.KNN.from_dataframe(countiesnick, k=8)
# Row-standardization
w.transform = 'R'


# In[ ]:


w


# In[ ]:




