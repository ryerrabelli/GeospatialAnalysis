#!/usr/bin/env python
# coding: utf-8

# # Pip install the non-common modules

# In[ ]:


get_ipython().system('pip install -U kaleido   # kaleido required for saving the plotly figures into static images')


# In[ ]:


get_ipython().system('pip install geopandas --quiet')
get_ipython().system('pip install geoplot --quiet')


# In[3]:


get_ipython().system('pip install watermark')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-d -m -v -p numpy,matplotlib,sklearn,pandas')


# # Set up

# In[ ]:


#@title ## Base imports
import os
import sys
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


# ## Load ENT df (specifically wide-type df) from csv file
# To understand what is meant by long type and wife type dataframes, see https://towardsdatascience.com/visualization-with-plotly-express-comprehensive-guide-eb5ee4b50b57

# In[ ]:


import pandas as pd
fips2county = pd.read_csv("data/fips2county.tsv", sep="\t", comment='#', dtype=str)
# The ent CSV file only contains the counties which are analyzable
df_orig = pd.read_csv("data/2022_04_10 ent initial output.csv", dtype={"FIPS": str})
# Merge with the fips 2 county standard data set
df_wide = pd.merge(left=df_orig, right=fips2county, how="left", left_on='FIPS', right_on='CountyFIPS')
# Insert a county "County, ST" col (i.e. "Freehold, NJ" or "Chicago, IL") for ease
df_wide.insert(1, "County_St", df_wide["CountyName"].astype(str) + ", " + df_wide["StateAbbr"].astype(str))
# Display with all the columns
with pd.option_context('display.max_rows', 3, 'display.max_columns', None): 
    display(df_wide)
    pass

loc_main = ["FIPS", "County",	"StateFIPS", "Total Medicare Payment Amount: 2019",	"CountyFIPS_3",	"CountyName",	"StateName",	"CountyFIPS",	"StateAbbr",	"STATE_COUNTY"]
#a=pd.merge(right=df_orig, left=fips2county, how="outer", right_on='FIPS', left_on='CountyFIPS')
#a=a.loc[:,loc_main]
#df_orig2=df_orig.loc[:,["FIPS","pop","Moran I score for ACS billing fraction","County"]]


# ## Convert wide df to long df - i.e. separate out the year columns into different rows

# In[ ]:


cols_to_keep = ["FIPS","County_St"]
col_categories = ["Total Number of Services:", "Total Medicare Payment Amount:", "% ASC Procedures:", "% ASC Billing:"]

df_longs = []

# Convert each type of category to long format in separate dataframes
for col_category in col_categories:
        df_long = df_wide.melt(id_vars=cols_to_keep, 
                               var_name="Year", 
                               value_vars=[f"{col_category} {year}" for year in range(2015, 2019 +1)], 
                               value_name=f"{col_category} in Year",
                               )
        df_long["Year"] = df_long["Year"].replace({ f"{col_category} {year}":f"{year}" for year in range(2015, 2019 +1)})
        df_longs.append(df_long)

# Merge the separate category dataframes
df_long = df_longs[0]
for ind in range(1,len(df_longs)):
    df_long = pd.merge(left=df_long, right=df_longs[ind], how="outer", on=(cols_to_keep+["Year"]) )

# Merge with the overall wide dataframe to keep those other values
df_long = pd.merge(left=df_long, 
                   right=df_wide.drop([f"{col_category} {year}" for year in range(2015, 2019 +1) for col_category in col_categories], axis=1), 
                   how="left", on=cols_to_keep)

display(df_long)


# # Analysis

# In[ ]:




fig = px.choropleth(df_wide, geojson=counties, locations='FIPS', 
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


fig = px.choropleth(df_long, geojson=counties, locations='FIPS', 
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


# In[ ]:




