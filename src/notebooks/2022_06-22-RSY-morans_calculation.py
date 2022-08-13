#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[177]:


countiesgeo = geopandas.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
countiesgeo = countiesgeo.sort_values(by="id")

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
#counties["features"][0]["id"]

counties2 = counties.copy()
counties2["features"] = sorted(counties2["features"], key=lambda d: d['id']) 
counties = counties2


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


# In[163]:


df_wide_geom = pd.merge(left=countiesgeo, right=df_wide, how="right", left_on='id', right_on='FIPS')
df_wide_simple_geom = pd.merge(left=countiesgeo, right=df_wide_simple, how="right", left_on='id', right_on='FIPS')


# In[164]:


df_wide_geom = df_wide_geom.set_index("FIPS").sort_index()
df_wide_simple_geom = df_wide_simple_geom.set_index("FIPS").sort_index()


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


# In[186]:


fig = px.choropleth(df_wide_simple_geom, geojson=df_wide_simple_geom.geometry, locations=df_wide_simple_geom.index, 
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


# In[192]:


w = weights.KNN.from_dataframe(df_wide_simple_geom, k=8)
# Row-standardization
w.transform = 'R'


# In[194]:


db = df_wide_simple_geom.copy()


# In[195]:


db['% ASC Billing_lag'] = weights.spatial_lag.lag_spatial(
    w, db['% ASC Billing']
)


# In[196]:


fig = px.choropleth(db, 
                    geojson=db.geometry, 
                    locations=db.index, 
                    color='% ASC Billing_lag',
                    color_continuous_scale="Viridis",
                    #range_color=(0, 12),
                    scope="usa",
                    )

fig.show()


#save_figure(fig,"choropleth-total")


# In[198]:


w.transform = 'R'
moran = esda.moran.Moran(db['% ASC Billing'], w)


# In[201]:


moran.I, moran.p_sim


# In[202]:


lisa = esda.moran.Moran_Local(db['% ASC Billing'], w)


# In[204]:


import seaborn
# Draw KDE line
ax = seaborn.kdeplot(lisa.Is)
# Add one small bar (rug) for each observation
# along horizontal axis
seaborn.rugplot(lisa.Is, ax=ax);


# In[207]:


from splot import esda as esdaplot
import matplotlib.pyplot as plt  # Graphics
from matplotlib import colors

# Set up figure and axes
f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()

                    # Subplot 1 #
            # Choropleth of local statistics
# Grab first axis in the figure
ax = axs[0]
# Assign new column with local statistics on-the-fly
db.assign(
    Is=lisa.Is
# Plot choropleth of local statistics
).plot(
    column='Is', 
    cmap='plasma', 
    scheme='quantiles',
    k=5, 
    edgecolor='white', 
    linewidth=0.1, 
    alpha=0.75,
    legend=True,
    ax=ax
)

                    # Subplot 2 #
                # Quadrant categories
# Grab second axis of local statistics
ax = axs[1]
# Plot Quandrant colors (note to ensure all polygons are assigned a
# quadrant, we "trick" the function by setting significance level to
# 1 so all observations are treated as "significant" and thus assigned
# a quadrant color
esdaplot.lisa_cluster(lisa, db, p=1, ax=ax);

                    # Subplot 3 #
                # Significance map
# Grab third axis of local statistics
ax = axs[2]
# 
# Find out significant observations
labels = pd.Series(
    1 * (lisa.p_sim < 0.05), # Assign 1 if significant, 0 otherwise
    index=db.index           # Use the index in the original data
# Recode 1 to "Significant and 0 to "Non-significant"
).map({1: 'Significant', 0: 'Non-Significant'})
# Assign labels to `db` on the fly
db.assign(
    cl=labels
# Plot choropleth of (non-)significant areas
).plot(
    column='cl', 
    categorical=True,
    k=2,
    cmap='Paired',
    linewidth=0.1,
    edgecolor='white',
    legend=True,
    ax=ax
)

                       
                    # Subplot 4 #
                    # Cluster map
# Grab second axis of local statistics
ax = axs[3]
# Plot Quandrant colors In this case, we use a 5% significance
# level to select polygons as part of statistically significant
# clusters
esdaplot.lisa_cluster(lisa, db, p=0.05, ax=ax);

                    # Figure styling #
# Set title to each subplot
for i, ax in enumerate(axs.flatten()):
    ax.set_axis_off()
    ax.set_title(
        [
            'Local Statistics', 
            'Scatterplot Quadrant', 
            'Statistical Significance', 
            'Moran Cluster Map'
        ][i], y=0
    )
# Tight layout to minimise in-betwee white space
f.tight_layout()

# Display the figure
plt.show()


# In[ ]:




