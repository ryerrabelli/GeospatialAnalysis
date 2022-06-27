#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import sys
import pandas as pd
import plotly.express as px


# In[8]:


import os
os.chdir("../..")


# In[10]:


os.getcwd()


# In[11]:


import geopandas
import rioxarray                 # Surface data manipulation
import xarray                    # Surface data manipulation
from pysal.explore import esda   # Exploratory Spatial analytics
from pysal.lib import weights    # Spatial weights
import contextily                # Background tiles


# In[155]:


fips2county = pd.read_csv("data/fips2county.tsv", sep="\t", comment='#', dtype=str).sort_values(by="CountyFIPS")
fips2countygeo = geopandas.read_file("data/plotly_usa_geojson-counties-fips.json").sort_values(by="id")
#fips2countygeo = geopandas.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")


# In[158]:


fips2county.head()


# In[161]:


df_orig.sort_values(by="FIPS").head()


# In[162]:


import pandas as pd

# The ent CSV file only contains the counties which are analyzable
df_orig = pd.read_csv("data/2022_04_10 ent initial output.csv", dtype={"FIPS": str}).sort_values(by="FIPS")
# Merge with the fips 2 county standard data set
df_wide = pd.merge(left=df_orig, right=fips2county, how="left", left_on='FIPS', right_on='CountyFIPS')
# Insert a county "County, ST" col (i.e. "Freehold, NJ" or "Chicago, IL") for ease
df_wide.insert(1, "County_St", df_wide["CountyName"].astype(str) + ", " + df_wide["StateAbbr"].astype(str))
# Display with all the columns
with pd.option_context('display.max_rows', 3, 'display.max_columns', None): 
    display(df_wide)
    pass

loc_simple = ["FIPS", "CountyName","StateAbbr", "% ASC Billing", "Moran I score for ACS billing fraction"]
df_wide_simple=df_wide[loc_simple]

loc_main = ["FIPS", "County",	"StateFIPS", "Total Medicare Payment Amount", "% ASC Procedures", "% ASC Billing",	"CountyFIPS_3",	"CountyName",	"StateName",	"CountyFIPS",	"StateAbbr",	"STATE_COUNTY"]
#a=pd.merge(right=df_orig, left=fips2county, how="outer", right_on='FIPS', left_on='CountyFIPS')
#a=a.loc[:,loc_main]
#df_orig2=df_orig.loc[:,["FIPS","pop","Moran I score for ACS billing fraction","County"]]


# In[ ]:





# In[13]:


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


# In[177]:


countiesgeo = geopandas.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
countiesgeo = countiesgeo.sort_values(by="id")


# In[178]:


countiesgeo.head()


# In[17]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[172]:


counties["features"][0]["id"]


# In[173]:


counties2 = counties.copy()
counties2["features"] = sorted(counties2["features"], key=lambda d: d['id']) 


# In[174]:


counties2


# In[137]:


df_wide_geom.head()


# In[163]:


df_wide_geom = pd.merge(left=countiesgeo, right=df_wide, how="right", left_on='id', right_on='FIPS')
df_wide_simple_geom = pd.merge(left=countiesgeo, right=df_wide_simple, how="right", left_on='id', right_on='FIPS')


# In[164]:


df_wide_geom = df_wide_geom.set_index("FIPS").sort_index()
df_wide_simple_geom = df_wide_simple_geom.set_index("FIPS").sort_index()


# In[146]:


df_wide_geom.geometry.to_json()


# In[179]:




fig = px.choropleth(df_wide_geom, geojson=counties2, locations=df_wide_geom.index, 
                    color='% ASC Procedures: 2019',
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
        'text': f"% ASC Procedures 2019",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


#save_figure(fig,"choropleth-total")


# In[182]:


fig = px.choropleth(df_wide_geom, geojson=df_wide_geom.geometry, locations=df_wide_geom.index, 
                    color='% ASC Billing',
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
        'text': f"% ASC Billing",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


#save_figure(fig,"choropleth-total")


# In[97]:


fig = px.choropleth(df_wide_geom,
                   geojson=df_wide_geom.to_json(),
                   locations=df_wide_geom.id,
                   color="% ASC Billing",
                    scope="usa",
                   #projection="mercator"
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.show()


# In[64]:


df_wide_geom.info()


# In[63]:


geo_df.info()


# In[119]:


geo_df.geometry


# In[121]:


import plotly.express as px
import geopandas as gpd

df = px.data.election()
geo_df = gpd.GeoDataFrame.from_features(
    px.data.election_geojson()["features"]
).merge(df, on="district") #.set_index("district")

fig = px.choropleth(geo_df,
                   geojson=geo_df.geometry,
                   locations=geo_df.index,
                   color="Joly",
                   projection="mercator")
fig.update_geos(fitbounds="locations", visible=True)
fig.show()


# In[96]:


geojson = px.data.election_geojson()
df = px.data.election()
geo_df = gpd.GeoDataFrame.from_features(
    px.data.election_geojson()["features"]
).merge(df, on="district").set_index("district")

fig = px.choropleth(geo_df, geojson=geojson, color="winner",
                    locations=geo_df.index, featureidkey="properties.district",
                    projection="mercator", hover_data=["Bergeron", "Coderre", "Joly"]
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[91]:


geojson


# In[ ]:




