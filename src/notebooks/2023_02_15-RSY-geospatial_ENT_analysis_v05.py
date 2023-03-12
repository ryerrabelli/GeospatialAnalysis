#!/usr/bin/env python
# coding: utf-8

# Created a notebook so it can be organized. Started Aug 12, 2022

# # Set up

# In[1]:


#@title ## Base imports
import os
import sys
import numpy as np
import scipy
import sklearn
import sklearn.linear_model
import pandas as pd
idx = pd.IndexSlice
import plotly
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import collections


import warnings
import requests
import urllib.request
import json
import copy   # to perform dict deep copy


# In[98]:


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


#@title ## Option 2) Clone project files from GitHub

get_ipython().system('git clone https://github.com/ryerrabelli/GeospatialAnalysis.git')

project_path_full = os.path.join("/content/","GeospatialAnalysis/")
sys.path.insert(1,project_path_full)
get_ipython().run_line_magic('cd', 'GeospatialAnalysis')
print(sys.path)


# # Helper functions

# In[79]:


colab_ip = get_ipython().run_line_magic('system', 'hostname -I   # uses colab magic to get list from bash')
colab_ip = colab_ip[0].strip()   # returns "172.28.0.12"
# Get most precent port name with !sudo lsof -i -P -n | grep LISTEN
colab_port = 9000                # could use 6000, 8080, or 9000

import requests
notebook_filename = filename = requests.get(f"http://{colab_ip}:{colab_port}/api/sessions").json()[0]["name"]

# Avoids scroll-in-the-scroll in the entire Notebook
def resize_colab_cell():
  display(IPython.display.Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 10000})'))
get_ipython().events.register('pre_run_cell', resize_colab_cell)


#@markdown # get_path_to_save()
def get_path_to_save(file_prefix="", save_filename:str=None, save_in_subfolder:str=None, extension="png", create_folder_if_necessary=True):
    save_path = ["outputs",
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
    df.to_pickle( get_path_to_save(save_filename=file_name, extension="pkl") )
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
        "SlopeSE": regress.stderr,
        "Intercept": regress.intercept, 
        "InterceptSE": regress.intercept_stderr,
        "R": regress.rvalue, 
        "p": regress.pvalue, 
         
        })


# # CMS API access - currently not used

# In[ ]:



# For the "Medicare Physician & Other Practitioners - by Provider and Service" database
# https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service

CMS_dataset_uuids = {
    2019:	"5fccd951-9538-48a7-9075-6f02b9867868",
    2018:	"02c0692d-e2d9-4714-80c7-a1d16d72ec66",
    2017:	"7ebc578d-c2c7-46fd-8cc8-1b035eba7218",
    2016:	"5055d307-4fb3-4474-adbb-a11f4182ee35",
    2015:	"0ccba18d-b821-47c6-bb55-269b78921637",
    }

# Column categories
#https://data.cms.gov/resources/medicare-physician-other-practitioners-by-provider-and-service-data-dictionary
"""Rndrng_NPI	TEXT
Rndrng_Prvdr_Last_Org_Name	TEXT
Rndrng_Prvdr_First_Name	TEXT
Rndrng_Prvdr_MI	TEXT
Rndrng_Prvdr_Crdntls	TEXT
Rndrng_Prvdr_Gndr	TEXT
Rndrng_Prvdr_Ent_Cd	TEXT
Rndrng_Prvdr_St1	TEXT
Rndrng_Prvdr_St2	TEXT
Rndrng_Prvdr_City	TEXT
Rndrng_Prvdr_State_Abrvtn	TEXT
Rndrng_Prvdr_State_FIPS	TEXT
Rndrng_Prvdr_Zip5	TEXT
Rndrng_Prvdr_RUCA	TEXT
Rndrng_Prvdr_RUCA_Desc	TEXT
Rndrng_Prvdr_Cntry	TEXT
Rndrng_Prvdr_Type	TEXT
Rndrng_Prvdr_Mdcr_Prtcptg_Ind	TEXT
HCPCS_Cd	TEXT
HCPCS_Desc	TEXT
HCPCS_Drug_Ind	TEXT
Place_Of_Srvc	TEXT
Tot_Benes	NUMERIC
Tot_Srvcs	NUMERIC
Tot_Bene_Day_Srvcs	NUMERIC
Avg_Sbmtd_Chrg	NUMERIC
Avg_Mdcr_Alowd_Amt	NUMERIC
Avg_Mdcr_Pymt_Amt	NUMERIC
Avg_Mdcr_Stdzd_Amt	NUMERIC"""   


# In[ ]:



def access_CMS_data(years, query, print_checkpoints=True, max_length=100000, return_pandas=True):
    records = {}
    if years is None:
        years = CMS_dataset_uuids.keys()
    for year in years:
        uuid = CMS_dataset_uuids[year]
        url_data = f"https://data.cms.gov/data-api/v1/dataset/{uuid}/data"
        url_stats = f"{url_data}/stats"
        query_stats = copy.deepcopy(query)
        stats_response = requests.get(url_stats, params=query_stats)
        if stats_response.status_code == 200:
            stats_response_json = stats_response.json()
            found_rows_ct = stats_response_json["found_rows"] 
            total_rows_ct = stats_response_json["total_rows"]
            if print_checkpoints: print(stats_response_json)

            query_offset = 0
            query_size = query["size"]
            records[year] = []
            while query_offset < found_rows_ct:
                query_data = copy.deepcopy(query)
                data_response = requests.get(url_data, params=query_data)
                if data_response.status_code == 200:
                    data_response_json = data_response.json()
                    # append lists
                    if print_checkpoints and query_offset>0: print("query_offset", query_offset)
                    records[year] = records[year] + data_response_json
                query_offset += query_size
    
    if return_pandas:
        import pandas as pd
        return pd.concat([pd.DataFrame.from_dict(year_of_records) for year_of_records in records.values()], keys=records.keys())
    else:
        # return as dict of list
        return records


# In[ ]:


uuid = CMS_dataset_uuids[2019]
query = {
    "column":"HCPCS_Cd,HCPCS_Desc,Tot_Benes", 
    #"group_by":"HCPCS_Cd",
    "offset":0, "size":10, "keyword":"30140"
    }
#url = f"https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/{uuid}/data?column=Rndrng_Prvdr_State_FIPS&offset=0&size=10"
url = f"https://data.cms.gov/data-api/v1/dataset/{uuid}/data"
#url = f"https://data.cms.gov/data-api/v1/dataset/{uuid}/data?column=Rndrng_Prvdr_State_FIPS&offset=0&size=10"
response = requests.get(url, params=query)

if response.status_code == 200:
    print(response.json())
    display(pd.DataFrame.from_dict(response.json()))


# In[ ]:


query = {
    #     "column":"HCPCS_Cd,HCPCS_Desc,Tot_Benes", 
    # 
    "column":"HCPCS_Cd,HCPCS_Desc,HCPCS_Drug_Ind,Place_Of_Srvc,Tot_Benes,Tot_Srvcs,Tot_Bene_Day_Srvcs,Avg_Sbmtd_Chrg,Avg_Mdcr_Alowd_Amt,Avg_Mdcr_Pymt_Amt,Avg_Mdcr_Stdzd_Am", 
    #"group_by":"HCPCS_Cd",
    "offset":0, "size":200, "keyword":"60240"
    }
records = access_CMS_data(None, query)


# In[ ]:


display(records)


# In[ ]:



save_df(records, "records_for_HCPCS=60240")


# # Procedures analysis

# ## Load ENT procedures df from csv file
# This is specifically a wide type df so it is one row per procedure with years as different columns.To understand what is meant by long type and wide type dataframes, see https://towardsdatascience.com/visualization-with-plotly-express-comprehensive-guide-eb5ee4b50b57

# The slope given in the csv file is actually the inverse slope. We need to either recalculate it or invert it. I will just recalculate all the regression values.

# In[149]:


df_procedures_orig = pd.read_csv("data/1_renamed/procedure_specific_data.csv",
                                 keep_default_na=False, # makes empty string cells still be interpreted as str 
                                 na_values=["nan", "NaN"],  # need to specify "NaN", not included by default
                                dtype={
                                    "Specialty": str,
                                    "Group": str,
                                    "HCPCS Code": str,
                                    "Total Number of Services": np.int64,
                                    **{f"Total Number of Services: {year}": np.int64 for year in range(2015,2019+1)},
                                    "% ASC Procedures": np.float64,
                                    "% ASC Billing": np.float64,
                                    })  # gets per healthcare code info


# ## Clean df and recalculate regression

# In[134]:


df_procedures_clean.columns.levels[2]
#df_procedures_clean['Total Medicare Payment'].columns.levels


# In[175]:


df_procedures_clean = df_procedures_orig.set_index(["Specialty","Group","HCPCS Code", "HCPCS Description"])

# Remove the "amount" word 
df_procedures_clean.columns = [col.replace("Total Medicare Payment Amount","Total Medicare Payment") for col in df_procedures_clean.columns]
# Make % words match the rest of the df
df_procedures_clean.columns = [col.replace("ASC Billing","ASC Payment") for col in df_procedures_clean.columns]
df_procedures_clean.columns = [col.replace("ASC Procedures","ASC Services") for col in df_procedures_clean.columns]

# Drop columns besides the individual year ones. Will recalculate the other ones as a quality assurance check.
df_procedures_clean = df_procedures_clean.drop(columns=[
    col for col in df_procedures_clean.columns 
    if (("slope" in col.lower() or "pearson" in col.lower() or ":" not in col) and "%" not in col)
    ] )

# Rename the columns so they can be split  easier. The 20 is the first two digits of the year columns (e.g. "2019") 
df_procedures_clean.columns = [col.replace(": ",": : ").replace(": 20","Annual: 20") for col in df_procedures_clean.columns]
df_procedures_clean.columns = [ (col.replace("% ASC ","ASC: ") + ": %" if "% ASC" in col else col) for col in df_procedures_clean.columns]

# Make Multiindex
df_procedures_clean.columns = pd.MultiIndex.from_tuples([tuple(col.split(": ")) if ":" in col else (col,"","") for col in df_procedures_clean.columns], names=["Category","Stat","Year"])
df_procedures_clean = df_procedures_clean[sorted(df_procedures_clean)]  # rearrange cols alphabetically

col_categories = df_procedures_clean.columns.levels[0]  #['ASC %', "Total Number of Services", "Total Medicare Payment Amount"]

# Make aggregates across the specialties and the groups
"""specialties = df_procedures_clean.index.unique(level="Specialty")
all_groups = df_procedures_clean.index.unique(level="Group")
df_procedures_clean.loc[("Total",None,"Total","Total")] = df_procedures_clean.sum()
for specialty in specialties:
    df_procedures_clean.loc[(specialty,None,"Total","Total")]=df_procedures_clean.loc[specialty].sum()
    groups = df_procedures_clean.loc[specialty].index.unique(level="Group")
    for group in groups:
        if df_procedures_clean.loc[(specialty,group)].shape[0] > 1:
            df_procedures_clean.loc[(specialty,group,"Total","Total")]=df_procedures_clean.loc[(specialty,group)].sum()"""

#df_procedures_clean = df_procedures_clean.sort_index()


# In[173]:


df_procedures_clean2 = df_procedures_clean.sort_index(level=[0,1,2,3])
df_procedures_clean2.index.is_monotonic_increasing

#df_procedures_clean.loc[(specialty,slice(None),slice("0","9"),slice(None))]
#df_procedures_clean.loc[idx[specialty,:,"31571":"31622"]]
#df_procedures_clean.loc[(specialty,"B","a","a"):tuple()]


# In[176]:


# Calculate regression and sum and mean from individual year later
df_procedures_recalc = df_procedures_clean.copy()

# Make aggregates across the specialties and the groups
specialties = df_procedures_recalc.index.unique(level="Specialty")
all_groups = df_procedures_recalc.index.unique(level="Group")
df_procedures_recalc.loc[("Total",None,"Total","Total")] = df_procedures_recalc.sum()
for specialty in specialties:
    df_procedures_recalc.loc[(specialty,None,"Total","Total")]=df_procedures_recalc.loc[specialty].sum()
    groups = df_procedures_recalc.loc[specialty].index.unique(level="Group")
    for group in groups:
        if df_procedures_recalc.loc[(specialty,group)].shape[0] > 1:
            df_procedures_recalc.loc[(specialty,group,"Total","Total")]=df_procedures_recalc.loc[(specialty,group)].sum()

for col_category in col_categories:
    if "Annual" in df_procedures_recalc[col_category].columns:
        new_df = df_procedures_recalc[(col_category,"Annual")].apply(calc_regression,axis=1, result_type="expand", args=(years,) )
        df_procedures_recalc[[(col_category,"Overall",new_col) for new_col in new_df.columns ]]=new_df
        #df_procedures_recalc[(col_category,"","Slope")]=df_procedures_recalc[(col_category,"Annual")].apply(calc_regression,axis=1)

for col_category in col_categories:
    if "Annual" not in df_procedures_recalc[col_category].columns:
        df_procedures_recalc[(col_category,"Payment","Mean")] =             df_procedures_recalc[(col_category,"Payment","%")] * df_procedures_recalc[("Total Medicare Payment","Overall","Mean")]
        df_procedures_recalc[(col_category,"Services","Mean")] =             df_procedures_recalc[(col_category,"Services","%")] * df_procedures_recalc[("Total Number of Services","Overall","Mean")]
        df_procedures_recalc[(col_category,"PaymentPerService","")] = df_procedures_recalc[(col_category,"Payment","Mean")] / df_procedures_recalc[(col_category,"Services","Mean")]
        df_procedures_recalc[(col_category,"Ratio","")] = df_procedures_recalc[(col_category,"Payment","%")] / df_procedures_recalc[(col_category,"Services","%")]

# rearrange cols alphabetically, but only by the first two elements of the each column's name tuple
# This allows the order of the newly added columns to remain relative to themselves, but be rearranged relative to the other columns
df_procedures_recalc = df_procedures_recalc[sorted(df_procedures_recalc.columns, key=(lambda x: x[0:2]))]  

#df_procedures = df_procedures.sort_values(by=("Total Number of Services","","Sum"), ascending=False)  # sort rows by volume 
df_procedures_recalc = df_procedures_recalc.sort_values(by=("Total Medicare Payment","Overall","Mean"), ascending=False)  # sort rows by volume 
df_procedures_recalc = df_procedures_recalc.sort_index()


# In[177]:


df_procedures_recalc


# ## Save recalculated procedures

# In[ ]:


with pd.option_context('display.float_format', '{:,.2f}'.format):
    display(df_procedures_recalc)

save_df(df_procedures_recalc, "df_procedures_recalc")


# ## Some stats

# In[ ]:





# ## Plot procedures in response to Reviewer 1's response (in round #2)

# In[ ]:


def customize_bar_chart(fig: plotly.graph_objs.Figure):
    fig.update_layout(
        template="simple_white",
        font=dict(
                family="Arial",
                size=16,
                color="black",
            ),
        xaxis=dict(
            zeroline=True,
            showgrid=True,
            mirror="ticks",
            gridcolor="#DDD",
            showspikes=True, spikemode="across", spikethickness=2, spikedash="solid"
        ),
        yaxis=dict(
            zeroline=True,
            showgrid=True,
            mirror="ticks",
            gridcolor="#DDD",
            showspikes=True, spikemode="across", spikethickness=2, spikedash="solid"
        ),

    )

var_labels = {
    "Total Medicare Payment-Overall-Mean": "Total Medicare Payment - 5yr mean ($/yr)",
    **{f"Total Medicare Payment-Annual-{yr}": f"Total Medicare Payment - {yr} ($/yr)" for yr in range(2015,2019+1)}
}
category_orders = {"Specialty": ["Facial plastics","Head & neck","Otology","Rhinology","Laryngology"]}


# In[ ]:


import plotly
plotly.__version__


# In[ ]:


rows = df_procedures_recalc.index.get_level_values(2) != "Total"
fig = px.bar(x=df_procedures_recalc.index.get_level_values(2)[rows], 
             y=df_procedures_recalc[("Total Medicare Payment","Overall","Mean")][rows], 
             color=df_procedures_recalc.index.get_level_values(0)[rows],
             category_orders=category_orders, labels=var_labels
             )
customize_bar_chart(fig)
fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.sort_values(by="Total Medicare Payment-Overall-Mean").reset_index()
df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b>")
fig = px.bar(df, 
             x="HCPCS", 
             y=("Total Medicare Payment-Overall-Mean"),
             color="Specialty", text_auto=".2s",
             category_orders=category_orders, labels=var_labels,
             )
customize_bar_chart(fig)
fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.sort_values(by="Total Medicare Payment-Overall-Mean").reset_index()
df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b>")
fig = px.bar(df, 
             x="HCPCS", 
             y=("Total Medicare Payment-Overall-Mean"),
             facet_col="Specialty",
             facet_col_wrap=2, text_auto='.2s', 
             category_orders=category_orders, labels=var_labels,
             )
fig.update_xaxes(matches=None)
fig.update_yaxes(matches=None)
#customize_bar_chart(fig)
fig.for_each_xaxis(lambda axis: axis.update(showticklabels=True))
fig.for_each_yaxis(lambda axis: axis.update(showticklabels=True))
fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.sort_values(by="Total Medicare Payment-Overall-Mean").reset_index()
df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b>")
fig = px.bar(df, 
             x="HCPCS", 
             y=[f"Total Medicare Payment-Annual-{yr}" for yr in range(2015,2019+1)],
             color="Specialty", text_auto='.2s',
             category_orders=category_orders, labels=var_labels,
             )
customize_bar_chart(fig)
fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.sort_values(by="Total Medicare Payment-Overall-Mean").reset_index()
df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b>")
fig = px.bar(df, 
             x="Specialty", 
             y="Total Medicare Payment-Overall-Mean",
             color="HCPCS Code",
             barmode="stack",
             category_orders=category_orders, labels=var_labels,
             text="HCPCS",
             )
fig.update_traces(textposition='inside')
customize_bar_chart(fig)
fig.show()


# In[ ]:


split_text = "<br>".join(textwrap.wrap('This is a very long title and it would be great to have it on three lines', 
                            width=30))
func = lambda string: "<br>".join(textwrap.wrap(string, width=50))
split_text


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.sort_values(by="Group")
df = df.reset_index()
stack = [f"X{ind}" for ind in range(100)]
stack.reverse()
df["Group"] = df["Group"].apply(lambda val: stack.pop() if val == "" else val)

df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b> " + df["HCPCS Description"]).apply(func)

fig = px.bar(df, 
             x="Specialty", 
             y="Total Medicare Payment-Overall-Mean",
             color="Group",
             barmode="stack",
             category_orders=category_orders, labels=var_labels,
             text="HCPCS",
             )
fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
fig.update_traces(textposition='inside')
customize_bar_chart(fig)
fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.reset_index()
df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b>")


fig = px.pie(df, names='Specialty', values='Total Medicare Payment-Overall-Mean')
fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.reset_index()
df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b>")


fig = px.pie(df, names='Specialty', 
             values='Total Medicare Payment-Overall-Mean'
)
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.reset_index()
df = df.sort_values(by="Group")
df["HCPCS"] = ("#" + df["HCPCS Code"] )
stack = [f"X{ind}" for ind in range(100)]
stack.reverse()
#df["Group"] = df["Group"].apply(lambda val: stack.pop() if val == "" else val)
#df["Group"] = df["Group"].apply(lambda val: None if val == "" else val)
df["Group"] = df["Group"].apply(lambda val: "Other" if val == "" else val)
fig = px.sunburst(df, path=["Specialty","Group","HCPCS"], values="Total Medicare Payment-Overall-Mean",
                  #color="Group",#color_continuous_scale='RdBu',
                  labels={"A":""},
                  )
fig.update_traces(textinfo="label+percent entry", selector=dict(type='sunburst'))

fig.update_traces(insidetextorientation="radial")
fig.show()


# In[ ]:


df = df_procedures_recalc.loc[df_procedures_recalc.index.get_level_values(2) != "Total"]
df.columns = ["-".join(col) for col in df.columns]
df = df.reset_index()
df["HCPCS"] = ("<b>#" + df["HCPCS Code"] + "</b>")
fig = px.sunburst(df, path=["Specialty","Group","HCPCS Code"], values="Total Medicare Payment-Overall-Mean")
fig.show()


# # County analysis

# ## Load data

# In[ ]:


# @title Load spatial coordinates of counties
# Below is necessary for plotting chloropleths. 
with urllib.request.urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[ ]:


# @title Load conversion df between FIPS code and county string
fips2county = pd.read_csv("data/fips2county.tsv", sep="\t", comment='#', dtype=str)


# In[ ]:


# @title Load our ENT df of all counties, their info, and the Moran's analysis
# The ent CSV file only contains the counties which are analyzable
df_counties_wide_orig = pd.read_csv("data/1_renamed/county_specific_data.csv", dtype={"FIPS": str})


# In[ ]:


df_counties_wide_orig.columns


# In[ ]:


# @title Merge with the fips 2 county standard data set
df_counties_wide = pd.merge(left=df_counties_wide_orig, right=fips2county, how="left", left_on='FIPS', right_on='CountyFIPS')
# Insert a county "County, ST" col (i.e. "Monmouth, NJ" or "Champaign, IL") for ease
df_counties_wide.insert(1, "County_St", df_counties_wide["CountyName"].astype(str) + ", " + df_counties_wide["StateAbbr"].astype(str))

cols_renamed={
    "Average Age": "Average Age (years)",
    'Percent Male': "% Male",
    'Percent Non-Hispanic White': "% Non-Hispanic White",
    'Percent African American': "% African American",
    'Percent Hispanic': "% Hispanic",
    'Percent Eligible for Medicaid': "% Eligible for Medicaid",
    'pct_poverty': "% Poverty",
    'median_house_income': "Median Household Income",
    "Pct_wthout_high_diploma": "% without High School Graduation",
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
    "Moran I score for ACS billing fraction":  "Moran I for ASC billing fraction",  # It is "ASC" not "ACS"
}
df_counties_wide = df_counties_wide.rename(columns=cols_renamed)


# In[ ]:


info_simple = ["FIPS", "CountyName","StateAbbr", "% ASC Billing"]
info_main = ["FIPS", "County",	"StateFIPS", "Total Medicare Payment Amount", "% ASC Procedures", "% ASC Billing",	"CountyFIPS_3",	"CountyName",	"StateName",	"CountyFIPS",	"StateAbbr",	"STATE_COUNTY", "Moran I for ASC billing fraction"]

df_counties_wide_simple=df_counties_wide[info_simple]
df_counties_wide_main=df_counties_wide[info_main]

# Display with all the columns
with pd.option_context('display.max_rows', 3, 'display.max_columns', None): 
    display(df_counties_wide_simple)


# ## Create analysis of which states were included

# In[ ]:


counties_with_datas_counts = df_counties_wide.groupby(["StateAbbr"])["StateAbbr"].count()
counties_with_datas_pop = df_counties_wide.groupby(["StateAbbr"])["Overall Population"].sum()
all_counties_counts = fips2county.groupby(["StateAbbr"])["StateAbbr"].count()


# In[ ]:


df_counties_by_state_and_Moran = df_counties_wide.groupby(["StateAbbr","Moran I for ASC billing fraction"])["StateAbbr"].count()
df_counties_by_state_and_Moran = pd.DataFrame(df_counties_by_state_and_Moran).rename(columns={"StateAbbr":"#"}).reset_index().pivot(index="StateAbbr",columns="Moran I for ASC billing fraction",values="#")
save_df(df_counties_by_state_and_Moran, "df_counties_by_state_and_Moran")


# In[ ]:


df_counties_by_state = pd.DataFrame({"Included":counties_with_datas_counts, "All Counties":all_counties_counts}, dtype="Int64")
df_counties_by_state["Ratio"] = df_counties_by_state["Included"]/df_counties_by_state["All Counties"]
df_counties_by_state["Included total population"] = counties_with_datas_pop
df_counties_by_state = df_counties_by_state.sort_values(by="Ratio",ascending=False)
display(df_counties_by_state)
save_df(df_counties_by_state, "df_counties_by_state")


# In[ ]:


df_counties_wide_main[["County","StateAbbr","Moran I for ASC billing fraction"]]


# ## Create long df from wide df- i.e. separate out the year columns into different rows

# In[ ]:


col_categories = ["Total Number of Services:", "Total Medicare Payment Amount:", "% ASC Procedures:", "% ASC Billing:"]
cols_to_keep = ["FIPS","County_St"]  # columns to keep in every subgroup so you can line up extra info later

# Create list of df's to combine later, each df is from melting of one col_category of columns
df_counties_longs = []

# Convert each type of col_category to long format in separate dataframes
for col_category in col_categories:
        df_counties_long = df_counties_wide.melt(id_vars=cols_to_keep, 
                               var_name="Year", 
                               value_vars=[f"{col_category} {year}" for year in range(2015, 2019 +1)], 
                               value_name=f"{col_category} in Year",
                               )
        df_counties_long["Year"] = df_counties_long["Year"].replace({ f"{col_category} {year}":f"{year}" for year in range(2015, 2019 +1)})
        df_counties_longs.append(df_counties_long)

# Merge the separate col_category dataframes
df_counties_long = df_counties_longs[0]
for ind in range(1,len(df_counties_longs)):
    df_counties_long = pd.merge(left=df_counties_long, right=df_counties_longs[ind], how="outer", on=(cols_to_keep+["Year"]) )

# Merge with the overall wide dataframe to keep those other values
df_counties_long = pd.merge(left=df_counties_long, 
                   right=df_counties_wide.drop([f"{col_category} {year}" for year in range(2015, 2019 +1) for col_category in col_categories], axis=1), 
                   how="left", on=cols_to_keep)


# In[ ]:


df_counties_long


# ## Set up for summaries and save sums

# In[ ]:


# sorted_moran_values = df_counties_wide["Moran I for ASC billing fraction"].unique()
sorted_moran_values = ["High-High","Low-Low","Low-High","High-Low","Non Significant"]  # list out specifically so you can get the order you want
sorted_moran_values_all = sorted_moran_values + ["All"]   #[pd.IndexSlice[:]]  # pd.IndexSlice[:]] represents all

moran_frequencies = df_counties_wide["Moran I for ASC billing fraction"].value_counts()[sorted_moran_values]


# In[ ]:


summable_groups = [col for col in df_counties_wide.columns if "total" in col.lower()]
summable_groups = summable_groups + ["Overall Population", "Medicare Beneficiaries Population"]
df_wide_sums = df_counties_wide.groupby("Moran I for ASC billing fraction")[summable_groups].sum()
df_wide_sums = df_wide_sums.assign(Counties=moran_frequencies)
df_wide_sums.loc["All"] = df_wide_sums.sum()

df_wide_sums = df_wide_sums[df_wide_sums.columns[::-1]]  # flip column order left-right to be more logical
with pd.option_context('display.float_format', '{:,.0f}'.format):
    display(df_wide_sums)

save_df(df_wide_sums, "df_wide_sums")


# In[ ]:


get_ipython().system('ls outputs/2022_11_20-RSY-geospatial_ENT_analysis_v03 ')


# ## Create summary data by Moran category

# In[ ]:


col_categories = ["Total Number of Services","Total Medicare Payment Amount", "% ASC Procedures", "% ASC Billing" ]

df_counties_with_slope = df_counties_wide.copy()
# Calculate regression and sum and mean from individual year later
for col_category in col_categories:
    new_df = df_counties_with_slope[ [col_category + ": " + str(yr) for yr in years] ].apply(calc_regression,axis=1, result_type="expand", args=(years,) )
    df_counties_with_slope[[col_category+": "+new_col for new_col in new_df.columns ]]=new_df
# To simplify, drop info for specific years unless it was "Mean" and "Slope" col_categories we just added
for col_category in col_categories:
    df_counties_with_slope = df_counties_with_slope.drop(columns=[col for col in df_counties_with_slope.columns if col_category in col and "Mean" not in col and "Slope" not in col])


df_counties_summary_dict = {}   # create a dict we will concatenate into a df later
# Options: 	[count, mean, std, min, 25%, 50%, 75%, max] assuming default percentiles argument
cols_to_show = ["10%","mean","90%"]
for possible_Moran_value in sorted_moran_values:
    df_counties_summary_dict[possible_Moran_value] = df_counties_with_slope[df_counties_with_slope["Moran I for ASC billing fraction"]==possible_Moran_value].describe(percentiles=[.1,.25,.5,.75,.9]).loc[cols_to_show]
df_counties_summary_dict["All"] = df_counties_with_slope.describe(percentiles=[.1,.25,.5,.75,.9]).loc[cols_to_show]

df_counties_summary = pd.concat(df_counties_summary_dict.values(), axis=0, keys=df_counties_summary_dict.keys())
for possible_Moran_value in sorted_moran_values:
    df_counties_summary.loc[(possible_Moran_value,cols_to_show[0]), "N"] = moran_frequencies[possible_Moran_value]

# Reorder into the sorted order we set above
df_counties_summary = df_counties_summary.loc[sorted_moran_values_all]


# ## Create a more presentable format
# Select out only the columns you want and rename the columns

# In[ ]:


df_counties_long.columns


# In[ ]:





# In[ ]:


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
    "Pct_wthout_high_diploma": "% without High School Graduation",
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
df_counties_summary_clean = df_counties_summary[key_cols.keys()]
df_counties_summary_clean = df_counties_summary_clean.rename(columns=key_cols).transpose()

with pd.option_context('display.float_format', '{:,.2f}'.format):
    display( df_counties_summary_clean )

save_df(df_counties_summary_clean, "df_counties_summary_clean")


# # Graph

# In[ ]:




