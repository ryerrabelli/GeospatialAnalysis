# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:13:58 2021

@author: nicho
"""
# use race2 environment for race calculations
import scipy.stats as stats
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import math
import glob
import libpysal as ps
from datetime import datetime, date


us_state_abbrev = {
    'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO',
    'Connecticut': 'CT','Delaware': 'DE','District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA',
    'Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA',
    'Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM',
    'New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Northern Mariana Islands':'MP','Ohio': 'OH',
    'Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI','Virginia': 'VA','Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'}

geoData = gpd.read_file('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')
geoData = geoData.loc[~geoData['STATE'].isin(['02', '78','72','60','66','69','15']) ]
geoData  = geoData.drop(['GEO_ID', 'COUNTY', 'NAME', 'LSAD'], axis=1)
statelist = geoData.STATE.unique()
geoData  = geoData.drop(['STATE'], axis=1)
geoData.rename(columns={'id':'FIPS'}, inplace=True)


npi_lookup =  pd.read_csv('npidata_3_15_2022.csv', usecols = ['NPI','Provider Business Practice Location Address Postal Code','Provider Business Practice Location Address State Name','Healthcare Provider Taxonomy Code_1','Healthcare Provider Taxonomy Code_2','Healthcare Provider Taxonomy Code_3'],dtype= "str")
npi_lookup.rename(columns={'Provider Business Practice Location Address Postal Code':'ZIP','Provider Business Practice Location Address State Name':'STATE'}, inplace=True)
npi_lookup['ZIP']= npi_lookup['ZIP'].str[:5] 
npi_lookup['ZIP'] = npi_lookup['ZIP'].astype(str).str.zfill(5)
npi_lookup.dropna(how= 'all',subset=['Healthcare Provider Taxonomy Code_1','Healthcare Provider Taxonomy Code_2','Healthcare Provider Taxonomy Code_3'],inplace = True)
npi_lookup.loc[npi_lookup['Healthcare Provider Taxonomy Code_1'].notnull(),'Healthcare Provider Taxonomy Code_1'] = npi_lookup.loc[npi_lookup['Healthcare Provider Taxonomy Code_1'].notnull(),'Healthcare Provider Taxonomy Code_1'].str[:10] 
npi_lookup.loc[npi_lookup['Healthcare Provider Taxonomy Code_2'].notnull(),'Healthcare Provider Taxonomy Code_2'] = npi_lookup.loc[npi_lookup['Healthcare Provider Taxonomy Code_2'].notnull(),'Healthcare Provider Taxonomy Code_2'].str[:10] 
npi_lookup.loc[npi_lookup['Healthcare Provider Taxonomy Code_3'].notnull(),'Healthcare Provider Taxonomy Code_3'] = npi_lookup.loc[npi_lookup['Healthcare Provider Taxonomy Code_3'].notnull(),'Healthcare Provider Taxonomy Code_3'].str[:10] 
code = ['207Y00000X']
npi_lookup =npi_lookup.loc[(npi_lookup['Healthcare Provider Taxonomy Code_1'].notnull() & npi_lookup['Healthcare Provider Taxonomy Code_1'].isin(code))|(npi_lookup['Healthcare Provider Taxonomy Code_2'].notnull() & npi_lookup['Healthcare Provider Taxonomy Code_2'].isin(code)) | (npi_lookup['Healthcare Provider Taxonomy Code_3'].notnull() & npi_lookup['Healthcare Provider Taxonomy Code_3'].isin(code))]
npi_lookup.loc[len(npi_lookup.index)] = ['0000000000','','','','',''] 

hcpcs = ['99213','G0463','31575','99214','99203','69210','31231','99232','99204','99222','99212','30520','99223','99221','31267','30140','99231','61782','31579','92504','31256','69436','99233','31253','99152',
'31525','31600','31237','99308','31541','31536','99202','38724','31535','99305','99215','31254','14060','60500','42415','99205','60220','31615','31571','60240','31259','31622','31257','31276','14302',
'14040','31238','99309','38510','30901','99304','99307','43200','69930','95811','64617','95004','15260','92511','95024','31502','31240','11642','69220','95810','21235','30903','14301','69433','69631',
'95940','42826','43191','99283','30802','30465','30930','11643','11042','15120','15004','G0500','14041','69801','15730','G0268','31526','92540','31288','42440','92557','20926','41120','30130','92537']
#npi_lookup['Provider Business Mailing Address Postal Code'] = npi_lookup['Provider Business Mailing Address Postal Code'].str[:5] 
df = pd.read_csv('Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2019.csv', dtype= "str", skiprows = [1],usecols=['National Provider Identifier',"Zip Code of the Provider",'Rndrng_Prvdr_St1', "HCPCS Code","Number of Services","Provider Type",'Avg_Mdcr_Alowd_Amt','Avg_Mdcr_Pymt_Amt','Tot_Bene_Day_Srvcs','Place of Service','HCPCS_Desc'])
df.rename(columns={'Rndrng_Prvdr_St1':'Street Address 1 of the Provider','Avg_Mdcr_Alowd_Amt':'Average Medicare Allowed Amount','Avg_Mdcr_Pymt_Amt':'Average Medicare Payment Amount','Tot_Bene_Day_Srvcs':'Number of Distinct Medicare Beneficiary/Per Day Services','HCPCS_Desc':'HCPCS Description'}, inplace=True)
df['Street Address 1 of the Provider'] = df['Street Address 1 of the Provider'].str.upper() 
df['Zip Code of the Provider'] = df['Zip Code of the Provider'].str[:5] 
df['Zip Code of the Provider'] = df['Zip Code of the Provider'].astype(str).str.zfill(5) 
df['tot'] = df['Street Address 1 of the Provider'] + df["Zip Code of the Provider"]
acs1 = df.loc[df["Provider Type"] =='Ambulatory Surgical Center','tot'].unique().tolist()
df.drop(['tot'],axis = 1,inplace= True)
df = df.loc[(df["Provider Type"].isin(['Otolaryngology','Ambulatory Surgical Center'])) & (df["HCPCS Code"].isin(hcpcs)) & (df["Place of Service"] == 'F')]
df['year'] = 2019
df1 = pd.read_csv('Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2018.csv', dtype= "str", skiprows = [1],usecols=['National Provider Identifier',"Zip Code of the Provider",'Street Address 1 of the Provider' ,"HCPCS Code","Number of Services","Provider Type",'Average Medicare Allowed Amount','Average Medicare Payment Amount','Number of Distinct Medicare Beneficiary/Per Day Services','Place of Service','HCPCS Description'])
df1 = df1.loc[(df1["Provider Type"].isin(['Otolaryngology','Ambulatory Surgical Center'])) & (df1["HCPCS Code"].isin(hcpcs)) & (df1["Place of Service"] == 'F')]
df1['Zip Code of the Provider'] = df1['Zip Code of the Provider'].str[:5] 
df1['Zip Code of the Provider'] = df1['Zip Code of the Provider'].astype(str).str.zfill(5) 
df1['Street Address 1 of the Provider'] = df1['Street Address 1 of the Provider'].str.upper() 
df1['year'] = 2018
df = pd.concat([df,df1], join='outer')
df2 = pd.read_csv('Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2017.csv', dtype= "str", skiprows = [1],usecols=['National Provider Identifier',"Zip Code of the Provider",'Street Address 1 of the Provider', "HCPCS Code","Number of Services","Provider Type",'Average Medicare Allowed Amount','Average Medicare Payment Amount','Number of Distinct Medicare Beneficiary/Per Day Services','Place of Service','HCPCS Description'])
df2 = df2.loc[(df2["Provider Type"].isin(['Otolaryngology','Ambulatory Surgical Center'])) & (df2["HCPCS Code"].isin(hcpcs)) & (df2["Place of Service"] == 'F')]
df2['Zip Code of the Provider'] = df2['Zip Code of the Provider'].str[:5] 
df2['Zip Code of the Provider'] = df2['Zip Code of the Provider'].astype(str).str.zfill(5) 
df2['Street Address 1 of the Provider'] = df2['Street Address 1 of the Provider'].str.upper()
df2['year'] = 2017
df = pd.concat([df,df2], join='outer')
df3 = pd.read_csv('Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2016.csv', dtype= "str", skiprows = [1],usecols=['National Provider Identifier',"Zip Code of the Provider",'Street Address 1 of the Provider', "HCPCS Code","Number of Services","Provider Type",'Average Medicare Allowed Amount','Average Medicare Payment Amount','Number of Distinct Medicare Beneficiary/Per Day Services','Place of Service','HCPCS Description'])
df3 = df3.loc[(df3["Provider Type"].isin(['Otolaryngology','Ambulatory Surgical Center'])) & (df3["HCPCS Code"].isin(hcpcs)) & (df3["Place of Service"] == 'F')]
df3['Zip Code of the Provider'] = df3['Zip Code of the Provider'].str[:5] 
df3['Zip Code of the Provider'] = df3['Zip Code of the Provider'].astype(str).str.zfill(5)
df3['Street Address 1 of the Provider'] = df3['Street Address 1 of the Provider'].str.upper() 
df3['year'] = 2016
df = pd.concat([df,df3], join='outer')
df4 = pd.read_csv('Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2015.csv', dtype= "str", skiprows = [1],usecols=['National Provider Identifier',"Zip Code of the Provider",'Street Address 1 of the Provider', "HCPCS Code","Number of Services","Provider Type",'Average Medicare Allowed Amount','Average Medicare Payment Amount','Number of Distinct Medicare Beneficiary/Per Day Services','Place of Service','HCPCS Description'])
df4 = df4.loc[(df4["Provider Type"].isin(['Otolaryngology','Ambulatory Surgical Center'])) & (df4["HCPCS Code"].isin(hcpcs)) & (df4["Place of Service"] == 'F')]
df4['Zip Code of the Provider'] = df4['Zip Code of the Provider'].str[:5] 
df4['Zip Code of the Provider'] = df4['Zip Code of the Provider'].astype(str).str.zfill(5)
df4['Street Address 1 of the Provider'] = df4['Street Address 1 of the Provider'].str.upper() 
df4['year'] = 2015
df = pd.concat([df,df4], join='outer')
#geoData  = geoData.drop(['STATE'], axis=1)
hcpcs_goodlist = df.loc[df["Provider Type"].isin(['Ambulatory Surgical Center']) & (df['HCPCS Code'] != '42826'),'HCPCS Code'].unique()

#hcpcs_goodlist = df.loc[df["Provider Type"].isin(['Ambulatory Surgical Center']),['HCPCS Code','HCPCS Description']].drop_duplicates()
backup = df.copy()
df = df.loc[df["HCPCS Code"].isin(hcpcs_goodlist)]

# filter out all of the cpt codes that are not ever billed to an ambulatory surgical center, before you run that function do the acs1 thing to expand the acs list and also document the level of expansion
df['Total Medicare Payment Amount'] = df['Average Medicare Payment Amount'].astype(float)*df['Number of Services'].astype(float)
for x in ['2019','2018','2017','2016','2015']:
    y1 = 'Total Number of Services: ' + x
    y2 = 'Total Medicare Payment Amount: ' + x
    #y3 = '% ACS Payment: ' + x
    df[[y1, y2]] = 0
    #df[[y1, y2,y3]] = 0
    df.loc[df['year'] == int(x),y1] = df.loc[df['year'] == int(x),'Number of Services'].astype(float) 
    df.loc[df['year'] == int(x),y2] = df.loc[df['year'] == int(x),'Total Medicare Payment Amount'].astype(float) 
    #df.loc[(df['year'] == int(x)) & (df['Provider Type'] == 'Ambulatory Surgical Center'),y3] = df.loc[(df['year'] == int(x)) & (df['Provider Type'] == 'Ambulatory Surgical Center'),'Total Medicare Payment Amount'].astype(float) 
df['Number of Services'] = df['Number of Services'].astype(float) 

df.drop(['year','Place of Service','Average Medicare Allowed Amount'],axis = 1,inplace= True) #'HCPCS Description'
df.rename(columns={'Zip Code of the Provider':'ZIP','Street Address 1 of the Provider':'ST_ADR','National Provider Identifier':'NPI','Number of Services':'Total Number of Services'}, inplace=True)

#summary stats

df2 = df.groupby(['HCPCS Code','HCPCS Description','Provider Type'], as_index=False).agg(sum)
df2[['% ASC Procedures','% ASC Billing']] = 0
df2['% ASC Procedures'] = df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center','Total Number of Services'].astype(float) 
df2['% ASC Billing'] = df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center','Total Medicare Payment Amount'].astype(float) 
df2 = df2.groupby(['HCPCS Code','HCPCS Description'], as_index=False).agg(sum)
df2['% ASC Procedures'] =  100*df2['% ASC Procedures']/df2['Total Number of Services']
df2['% ASC Billing'] = 100*df2['% ASC Billing']/df2['Total Medicare Payment Amount']
df2.fillna(0,inplace = True)
for x in ['2019','2018','2017','2016','2015']:
    y1 = 'Total Medicare Payment Amount: ' + x
    y2 = '% ACS Payment: ' + x
    df2[y2] =100*df2[y2]/df2[y1]

df2.fillna(0,inplace = True)
axisvalues = [1,2,3,4,5]

def calc_slope(row):
    a = stats.linregress(axisvalues, y=row)
    return a.slope 
def calc_rvalue(row):
    a = stats.linregress(axisvalues, y=row)
    return a.rvalue

df2['Total Medicare Payment Amount: Slope'] = df2[['Total Medicare Payment Amount: 2015','Total Medicare Payment Amount: 2016','Total Medicare Payment Amount: 2017','Total Medicare Payment Amount: 2018','Total Medicare Payment Amount: 2019']].apply(calc_slope,axis=1)
df2['Total Medicare Payment Amount: Pearson Coef'] = df2[['Total Medicare Payment Amount: 2015','Total Medicare Payment Amount: 2016','Total Medicare Payment Amount: 2017','Total Medicare Payment Amount: 2018','Total Medicare Payment Amount: 2019']].apply(calc_rvalue,axis=1)
df2['Total Number of Services: Slope'] = df2[['Total Number of Services: 2015','Total Number of Services: 2016','Total Number of Services: 2017','Total Number of Services: 2018','Total Number of Services: 2019']].apply(calc_slope,axis=1)
df2['Total Number of Services: Pearson Coef'] = df2[['Total Number of Services: 2015','Total Number of Services: 2016','Total Number of Services: 2017','Total Number of Services: 2018','Total Number of Services: 2019']].apply(calc_rvalue,axis=1)
df2['% ASC Billing: Slope'] = df2[['% ACS Payment: 2015','% ACS Payment: 2016','% ACS Payment: 2017','% ACS Payment: 2018','% ACS Payment: 2019']].apply(calc_slope,axis=1)
df2['% ASC Billing: Pearson Coef'] = df2[['% ACS Payment: 2015','% ACS Payment: 2016','% ACS Payment: 2017','% ACS Payment: 2018','% ACS Payment: 2019']].apply(calc_rvalue,axis=1)


df2.to_csv("sums and slopes ent2.csv",index=False)
df3 = df2[['HCPCS Code', 'HCPCS Description', 'Total Number of Services','Total Medicare Payment Amount', '% ASC Procedures', '% ASC Billing',
 'Total Medicare Payment Amount: Slope','Total Medicare Payment Amount: Pearson Coef','Total Number of Services: Slope','Total Number of Services: Pearson Coef', '% ASC Billing: Slope','% ASC Billing: Pearson Coef']]
df3['ACS % Billing to % Procedure Ratio'] = df3['% ASC Billing']/df3['% ASC Procedures']
df3.to_csv("ent table.csv",index=False)


zip_county_cross = pd.read_csv("zip_county_cross.csv",usecols=["ZIP", "FIPS",'tot_ratio'], dtype=str)
zip_county_cross['FIPS'] = zip_county_cross['FIPS'].str.zfill(5) 
zip_county_cross['ZIP'] = zip_county_cross['ZIP'].str.zfill(5) 
zip_county_cross['tot_ratio'] = zip_county_cross['tot_ratio'].astype(float)
zip_county_cross = zip_county_cross.loc[zip_county_cross.groupby(['ZIP'])['tot_ratio'].idxmax()]
df = df.merge(zip_county_cross ,how = 'inner', on='ZIP') 

#df.loc[df["Provider Type"] =='Ambulatory Surgical Center','NPI'] = '0000000000'
df2 = df.copy()
#df2 = df2.merge(npi_lookup[['NPI']], how = 'inner',on ='NPI')
df2 = df2.groupby(['FIPS',"Provider Type"], as_index=False).agg(sum)

for x in ['2019','2018','2017','2016','2015']:
    y1 = '% ASC Procedures: ' + x
    y2 = '% ASC Billing: ' + x
    z1 = 'Total Number of Services: ' + x
    z2 = 'Total Medicare Payment Amount: ' + x
    df2[[y1, y2]] = 0
    df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center',y1] = df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center',z1].astype(float) 
    df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center',y2] = df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center',z2].astype(float) 
df2[['% ASC Procedures','% ASC Billing']] = 0
df2['% ASC Procedures'] = df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center','Total Number of Services'].astype(float) 
df2['% ASC Billing'] = df2.loc[df2['Provider Type'] == 'Ambulatory Surgical Center','Total Medicare Payment Amount'].astype(float) 

df2 = df2.groupby(['FIPS'], as_index=False).agg(sum)


for x in ['2019','2018','2017','2016','2015']:
    y1 = '% ASC Procedures: ' + x
    y2 = '% ASC Billing: ' + x
    z1 = 'Total Number of Services: ' + x
    z2 = 'Total Medicare Payment Amount: ' + x
    df2[y1] =100*df2[y1]/df2[z1]
    df2[y2] =100*df2[y2]/df2[z2] 
df2['% ASC Procedures'] =  100*df2['% ASC Procedures']/df2['Total Number of Services']
df2['% ASC Billing'] = 100*df2['% ASC Billing']/df2['Total Medicare Payment Amount']
df2.replace(np.nan, 0,inplace = True)

final = df2.copy()
final['Total Medicare Payment Amount: Slope'] = final[['Total Medicare Payment Amount: 2015','Total Medicare Payment Amount: 2016','Total Medicare Payment Amount: 2017','Total Medicare Payment Amount: 2018','Total Medicare Payment Amount: 2019']].apply(calc_slope,axis=1)
final['Total Number of Services: Slope'] = final[['Total Number of Services: 2015','Total Number of Services: 2016','Total Number of Services: 2017','Total Number of Services: 2018','Total Number of Services: 2019']].apply(calc_slope,axis=1)
final['% ASC Billing: Slope'] = final[['% ASC Billing: 2015','% ASC Billing: 2016','% ASC Billing: 2017','% ASC Billing: 2018','% ASC Billing: 2019']].apply(calc_slope,axis=1)
final['% ASC Procedures: Slope'] = final[['% ASC Procedures: 2015','% ASC Procedures: 2016','% ASC Procedures: 2017','% ASC Procedures: 2018','% ASC Procedures: 2019']].apply(calc_slope,axis=1)
final.fillna(0,inplace = True)



import openpyxl

columns = ['State and County FIPS Code','Beneficiaries with Part A and Part B','Average Age','Percent Male','Percent Non-Hispanic White','Percent African American','Percent Hispanic','Percent Eligible for Medicaid','Average HCC Score','Emergency Department Visits per 1000 Beneficiaries','Hospital Readmission Rate','Procedures Per Capita Standardized Costs','Procedure Events Per 1000 Beneficiaries']
medi=pd.read_excel("County All Table 2019.xlsx", skiprows = [0,2], sheet_name="State_county 2019",engine='openpyxl', usecols =columns, dtype= "str")
for year in ['2018','2017','2016','2015']:
    medi = pd.concat([medi,pd.read_excel("County All Table 2019.xlsx", skiprows = [0,2], sheet_name="State_county 2019".replace('2019',year),engine='openpyxl', usecols =columns, dtype= "str")], join='outer')
medi = medi.dropna(subset=['State and County FIPS Code'])
medi['State and County FIPS Code'] = medi['State and County FIPS Code'].astype(str).str.zfill(5) 
medi[columns[1:]] = medi[columns[1:]].replace({'\*': np.NaN,' %':''}, regex=True).apply(pd.to_numeric, args=('coerce',))
medi = medi.dropna(subset=['Beneficiaries with Part A and Part B'])
medi = medi.groupby(['State and County FIPS Code'], as_index=False).agg(np.mean)
medi = medi.fillna(0)
medi.rename(columns={'State and County FIPS Code':'FIPS'}, inplace=True)
gdf =  final.merge(medi, on='FIPS', how='inner')

gdf = geoData.merge(gdf, on='FIPS', how='inner')


def disease(filename,year):
    df = pd.read_csv(filename, usecols=['FIPS','Alcohol Abuse','Drug Abuse',"Alzheimers","Depression",'Schizo_othr_psych','COPD','Chronic Kidney Disease','Osteoporosis','Stroke','Diabetes','Asthma','Arthritis','Hypertension','Heart Failure','Ischemic Heart Disease'])
    df.replace('  ', np.NaN, inplace=True)
    df.dropna(subset=['FIPS'],inplace = True)
    df.replace('* ', np.NaN, inplace=True)
    df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)
    for col in df.columns[1:]:
        print(col)
        placeholder = df[df.columns[1:]].dropna()
        m = 5*placeholder.loc[placeholder[col]>0,col].min()/12
        df.loc[~(df[col]>=0),col] = m 
    df.rename(columns={'Alcohol Abuse':'Alcohol Abuse'+year, "Alzheimers": "Alzheimers" + year,"Depression": "Depression"+ year,'Drug Abuse':'Drug Abuse' + year,'Schizo_othr_psych': 'Schizo_othr_psych' + year,'COPD': 'COPD'+ year,'Chronic Kidney Disease': 'Chronic Kidney Disease' +year,'Osteoporosis': 'Osteoporosis' + year,'Stroke' : 'Stroke' + year,'Diabetes': 'Diabetes' + year ,'Asthma' : 'Asthma' + year, 'Arthritis':'Arthritis' + year, 'Hypertension':'Hypertension' + year,'Heart Failure':'Heart Failure' + year,'Ischemic Heart Disease':'Ischemic Heart Disease' + year}, inplace=True)
    return df
def disease2019(filename,year):
    df = pd.read_csv(filename, usecols=['FIPS',"Alzheimers","Depression",'Schizo_othr_psych','COPD','Chronic Kidney Disease','Osteoporosis','Stroke','Diabetes','Asthma','Arthritis','Hypertension','Heart Failure','Ischemic Heart Disease'])
    df.replace('  ', np.NaN, inplace=True)
    df.dropna(subset=['FIPS'],inplace = True)
    df.replace('* ', np.NaN, inplace=True)
    df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)
    for col in df.columns[1:]:
        print(col)
        placeholder = df[df.columns[1:]].dropna()
        m = 5*placeholder.loc[placeholder[col]>0,col].min()/12
        df.loc[~(df[col]>=0),col] = m 
    df.rename(columns={"Alzheimers": "Alzheimers" + year,"Depression": "Depression"+ year,'Schizo_othr_psych': 'Schizo_othr_psych' + year,'COPD': 'COPD'+ year,'Chronic Kidney Disease': 'Chronic Kidney Disease' +year,'Osteoporosis': 'Osteoporosis' + year,'Stroke' : 'Stroke' + year,'Diabetes': 'Diabetes' + year ,'Asthma' : 'Asthma' + year,'Arthritis':'Arthritis' + year,'Hypertension':'Hypertension' + year,'Heart Failure':'Heart Failure' + year,'Ischemic Heart Disease':'Ischemic Heart Disease' + year}, inplace=True)
    return df
df = disease('chronic_disease_2015.csv','15')
df = pd.merge(df,disease('chronic_disease_2016.csv','16'),how = 'inner', on='FIPS')
df = pd.merge(df,disease('chronic_disease_2017.csv','17'),how = 'inner', on='FIPS')
df = pd.merge(df,disease('chronic_disease_2018.csv','18'),how = 'inner', on='FIPS')
df = pd.merge(df,disease2019('chronic disease 2019.csv','19'),how = 'inner', on='FIPS')
for col in ["Alzheimers","Depression",'Alcohol Abuse','Drug Abuse', 'Schizo_othr_psych','COPD','Chronic Kidney Disease','Osteoporosis','Stroke','Diabetes','Asthma','Arthritis','Hypertension','Heart Failure','Ischemic Heart Disease']:
    if ((col == 'Alcohol Abuse' ) | (col == 'Drug Abuse' )):
        df[col] = (df[col+'15'] + df[col+'16'] + df[col+'17'] + df[col+'18'])/4
        df.drop([col+'15',col+'16',col+'17',col+'18'], axis=1, inplace=True)
    else:
        df[col] = (df[col+'15'] + df[col+'16'] + df[col+'17'] + df[col+'18'] + df[col+'19'])/5
        df.drop([col+'15',col+'16',col+'17',col+'18',col+'19'], axis=1, inplace=True)


education = pd.read_csv("Education.csv", usecols = ["fips","2013_Rural_urban_cont_code","Pct_wthout_high_diploma","Pct_wth_high_diploma","Pct_wth_some_coll","Pct_wth_coll_degree"])
education = education.dropna()
education['fips'] = education['fips'].astype(str).str.zfill(5)
education.rename(columns={'fips':'FIPS'}, inplace=True)


pov15 = pd.read_csv("est15all.csv", usecols=["State FIPS Code",'County FIPS Code', "pct_poverty15","median_house_income15"])
pov15['FIPS']= pov15["State FIPS Code"].astype(str).str.zfill(2) + pov15["County FIPS Code"].astype(str).str.zfill(3)
pov15.drop(["State FIPS Code",'County FIPS Code'], axis=1, inplace=True)
pov15.loc[pov15["pct_poverty15"] == '.',["pct_poverty15","median_house_income15"]] = np.NaN
pov15[["pct_poverty15","median_house_income15"]]= pov15[["pct_poverty15","median_house_income15"]].astype(float)
pov15 = pov15.dropna()

pov16 = pd.read_csv("est16all.csv", usecols=["State FIPS Code",'County FIPS Code', "pct_poverty16","median_house_income16"])
pov16['FIPS']= pov16["State FIPS Code"].astype(str).str.zfill(2) + pov16["County FIPS Code"].astype(str).str.zfill(3)
pov16.drop(["State FIPS Code",'County FIPS Code'], axis=1, inplace=True)
pov16.loc[pov16["pct_poverty16"] == '.',["pct_poverty16","median_house_income16"]] = np.NaN
pov16[["pct_poverty16","median_house_income16"]]= pov16[["pct_poverty16","median_house_income16"]].astype(float)
pov16 = pov16.dropna()

pov17 = pd.read_csv("est17all.csv", usecols=["State FIPS Code",'County FIPS Code', "pct_poverty17","median_house_income17"])
pov17['FIPS']= pov17["State FIPS Code"].astype(str).str.zfill(2) + pov17["County FIPS Code"].astype(str).str.zfill(3)
pov17.drop(["State FIPS Code",'County FIPS Code'], axis=1, inplace=True)
pov17.loc[pov17["pct_poverty17"] == '.',["pct_poverty17","median_house_income17"]] = np.NaN
pov17[["pct_poverty17","median_house_income17"]]= pov17[["pct_poverty17","median_house_income17"]].astype(float)
pov17 = pov17.dropna()

pov18 = pd.read_csv("est18all.csv", usecols=["State FIPS Code",'County FIPS Code', "pct_poverty18","median_house_income18"])
pov18['FIPS']= pov18["State FIPS Code"].astype(str).str.zfill(2) + pov18["County FIPS Code"].astype(str).str.zfill(3)
pov18.drop(["State FIPS Code",'County FIPS Code'], axis=1, inplace=True)
pov18.loc[pov18["pct_poverty18"] == '.',["pct_poverty18","median_house_income18"]] = np.NaN
pov18[["pct_poverty18","median_house_income18"]]= pov18[["pct_poverty18","median_house_income18"]].astype(float)
pov18 = pov18.dropna()

pov19 = pd.read_csv("est19all.csv", usecols=["State FIPS Code",'County FIPS Code', "pct_poverty19","median_house_income19"])
pov19['FIPS']= pov19["State FIPS Code"].astype(str).str.zfill(2) + pov19["County FIPS Code"].astype(str).str.zfill(3)
pov19.drop(["State FIPS Code",'County FIPS Code'], axis=1, inplace=True)
pov19.loc[pov19["pct_poverty19"] == '.',["pct_poverty19","median_house_income19"]] = np.NaN
pov19[["pct_poverty19","median_house_income19"]]= pov19[["pct_poverty19","median_house_income19"]].astype(float)
pov19 = pov19.dropna()

pov = pd.merge(pov15,pov16,how = 'inner', on='FIPS')
pov = pd.merge(pov,pov17,how = 'inner', on='FIPS')
pov = pd.merge(pov,pov18,how = 'inner', on='FIPS')
pov = pd.merge(pov,pov19,how = 'inner', on='FIPS')
pov['pct_poverty'] = (pov['pct_poverty15'] + pov['pct_poverty16'] + pov['pct_poverty17'] + pov['pct_poverty18']+ pov['pct_poverty19'])/5
pov['median_house_income'] = (pov['median_house_income15'] + pov['median_house_income16'] + pov['median_house_income17'] + pov['median_house_income18'] + pov['median_house_income19'])/5
pov = pov[['FIPS','pct_poverty','median_house_income']]

metro = pd.read_csv("metro.csv", usecols = ["FIPS","metro"])
metro = metro.dropna()
metro['FIPS'] = metro['FIPS'].astype(str).str.zfill(5)

# i dont know why you did this but 1-6 matches up to 2019 -2014
migrane = geoData.copy()
migrane['migrane'] = 0 
for i in ['1','2','3','4','5']:
    holder = pd.read_csv("migrane all years.csv", usecols = ['FIPS'+i,'migrane'+i],dtype= "str")
    holder = holder.dropna()
    holder['FIPS'+i] = holder['FIPS'+i].astype(str).str.zfill(5)
    holder.rename(columns={'FIPS'+i:'FIPS'}, inplace=True)
    migrane = pd.merge(migrane, holder, how = 'left',on ='FIPS')
    migrane['migrane'] = migrane['migrane'] + migrane['migrane'+i].astype(float)
migrane['migrane'] =migrane['migrane']/5
migrane = migrane[['FIPS','migrane']]

tobo = geoData.copy()
tobo['tabacco'] = 0
for i in ['1','2','3','4','5']:
    holder = pd.read_csv("tobacco all years.csv", usecols = ['FIPS'+i,'tabacco'+i],dtype= "str")
    holder = holder.dropna()
    holder['FIPS'+i] = holder['FIPS'+i].astype(str).str.zfill(5)
    holder.rename(columns={'FIPS'+i:'FIPS'}, inplace=True)
    tobo = pd.merge(tobo, holder, how = 'left',on ='FIPS')
    tobo['tabacco'] = tobo['tabacco'] + tobo['tabacco'+i].astype(float)
tobo['tabacco'] =tobo['tabacco']/5
tobo = tobo[['FIPS','tabacco']]

obo = geoData.copy()
obo['obesity'] = 0
for i in ['1','2','3','4','5']:
    holder = pd.read_csv("obesity all years.csv", usecols = ['FIPS'+i,'obesity'+i],dtype= "str")
    holder = holder.dropna()
    holder['FIPS'+i] = holder['FIPS'+i].astype(str).str.zfill(5)
    holder.rename(columns={'FIPS'+i:'FIPS'}, inplace=True)
    obo = pd.merge(obo, holder, how = 'left',on ='FIPS')
    obo['obesity'] = obo['obesity'] + obo['obesity'+i].astype(float)
obo['obesity'] =obo['obesity']/5
obo = obo[['FIPS','obesity']]

fibro = geoData.copy()
fibro['fibro'] = 0
for i in ['1','2','3','4','5']:
    holder = pd.read_csv("fibro all years.csv", usecols = ['FIPS'+i,'fibro'+i],dtype= "str")
    holder = holder.dropna()
    holder['FIPS'+i] = holder['FIPS'+i].astype(str).str.zfill(5)
    holder.rename(columns={'FIPS'+i:'FIPS'}, inplace=True)
    fibro = pd.merge(fibro, holder, how = 'left',on ='FIPS')
    fibro['fibro'] = fibro['fibro'] + fibro['fibro'+i].astype(float)
fibro['fibro'] =fibro['fibro']/5
fibro = fibro[['FIPS','fibro']]

uninsured = pd.read_csv("uninsured.csv", usecols = ["FIPS",'year',"pct_uninsured"])
uninsured = uninsured.dropna()
uninsured['FIPS'] = uninsured['FIPS'].astype(str).str.zfill(5)
uninsured = uninsured.loc[uninsured['year'] >= 2015]
uninsured=uninsured.groupby(['FIPS'],as_index=False).mean()
uninsured.drop(['year'], axis=1, inplace=True)

unemp = pd.read_csv("Unemployment.csv", usecols = ["FIPS","Unemployment_rate_2015","Unemployment_rate_2016","Unemployment_rate_2017","Unemployment_rate_2018","Unemployment_rate_2019"])# this is just for getting location for now
unemp = unemp.dropna()
unemp['FIPS'] = unemp['FIPS'].astype(str).str.zfill(5)
unemp['unemployment'] = (unemp["Unemployment_rate_2015"] + unemp["Unemployment_rate_2016"] + unemp["Unemployment_rate_2017"] + unemp["Unemployment_rate_2018"]+ unemp["Unemployment_rate_2019"])/5
unemp.drop(["Unemployment_rate_2015","Unemployment_rate_2016","Unemployment_rate_2017","Unemployment_rate_2018","Unemployment_rate_2019"], axis=1, inplace=True)

pop = pd.read_csv("US_county_census_est_race_eth_2010_2019.csv",usecols = ["fips",'pop','year'])
pop = pop.loc[pop['year'] >= 2015]
pop = pop.dropna()
pop['fips'] = pop['fips'].astype(str).str.zfill(5)
pop=pop.groupby(['fips'],as_index=False).mean()
pop.rename(columns={'fips':'FIPS'}, inplace=True)
pop.drop(["year"], axis=1, inplace=True)


gdf = pd.merge(gdf,metro,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,pov,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,pop,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,education,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,unemp,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,uninsured,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,fibro,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,tobo,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,obo,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,migrane,how = 'inner', on='FIPS')
gdf = pd.merge(gdf,df,how = 'inner', on='FIPS')
gdf = gpd.GeoDataFrame(gdf)

gdf = gdf.fillna(0)
gdf['Population Density'] = gdf.apply(lambda row: row['pop']/(row.CENSUSAREA), axis=1)
gdf['Medicare Population Density'] = gdf.apply(lambda row: row['Beneficiaries with Part A and Part B']/(row.CENSUSAREA), axis=1)
gdf['Urban'] = 0
gdf.loc[gdf['2013_Rural_urban_cont_code'] == 1,'Urban'] = 1
gdf = gdf.fillna(0)
gdf2 = gdf.loc[gdf['Beneficiaries with Part A and Part B'] > 100]

log_list =  ['Population Density','Medicare Population Density','median_house_income']
for col in log_list:
    gdf2["Log of "+col] = np.log10(gdf2[col]+gdf2.loc[gdf2[col] > 0.01,col].min()/2)


gdf2['ENT ASC Procedures per 10k Medicare Members'] = 10000*gdf2['2015to2019']/gdf2['Beneficiaries with Part A and Part B']

gdf2.to_file("ent asc counties.json", driver="GeoJSON")
gdf.to_file("ent redo2.json", driver="GeoJSON")

#numbers are fucky in the slopes, double check them



def weighted_sd(input_df,wcol, col):
    weights = input_df[wcol]
    vals = input_df[col]
    numer = np.sum(weights * (vals - vals.mean())**2)
    denom = ((vals.count()-1)/vals.count())*np.sum(weights)
    return round(np.sqrt(numer/denom),2)

def anova_to_csv(df,cols,weightcol):
    df1 = pd.DataFrame(np.zeros((10, len(cols))))
    df1.columns = cols
    dfscale = df.copy()
    for col in cols:
        df1[col][0] = round(sum(df.loc[df.LISA_CL == 1,weightcol]*df.loc[df.LISA_CL == 1,col])/df.loc[df.LISA_CL == 1,weightcol].sum(),2)
        df1[col][1] = weighted_sd(df.loc[df.LISA_CL == 1],weightcol, col)
        df1[col][2] = round(sum(df.loc[df.LISA_CL == 2,weightcol]*df.loc[df.LISA_CL == 2,col])/df.loc[df.LISA_CL == 2,weightcol].sum(),2)
        df1[col][3] = weighted_sd(df.loc[df.LISA_CL == 2],weightcol, col)
        df1[col][4] = round(sum(df.loc[df.LISA_CL == 3,weightcol]*df.loc[df.LISA_CL == 3,col])/df.loc[df.LISA_CL == 3,weightcol].sum(),2)
        df1[col][5] = weighted_sd(df.loc[df.LISA_CL == 3],weightcol, col)
        df1[col][6] = round(sum(df.loc[df.LISA_CL == 4,weightcol]*df.loc[df.LISA_CL == 4,col])/df.loc[df.LISA_CL == 4,weightcol].sum(),2)
        df1[col][7] = weighted_sd(df.loc[df.LISA_CL == 4],weightcol, col)
        #transform that column based on minimizing skew
        if (len(dfscale.loc[dfscale[col]<0])==0):
            if(abs(dfscale[col].apply(lambda x: np.log10(x + 0.0001)).skew()) < abs(dfscale[col].skew())):
                dfscale[col]= dfscale[col].apply(lambda x: np.log10(x + 0.0001))
            try:
                if (abs(stats.skew(stats.boxcox(dfscale[col])[0])) < abs(dfscale[col].skew())):
                    dfscale[col] = stats.boxcox(dfscale[col])[0]
            except Exception:
                pass
        print(dfscale[col].skew())
        f,p= stats.stats.f_oneway(dfscale.loc[dfscale.LISA_CL == 1,col],dfscale.loc[dfscale.LISA_CL == 2,col],dfscale.loc[dfscale.LISA_CL == 3,col],dfscale.loc[dfscale.LISA_CL == 4,col])
        df1[col][8] = p
        f,p= stats.stats.ttest_ind(dfscale.loc[dfscale.LISA_CL == 1,col],dfscale.loc[dfscale.LISA_CL == 2,col])
        df1[col][9] = p
    return df1
clust = gpd.read_file('ent redo2anova2.geojson')
clust.drop(['CENSUSAREA','geometry'],axis=1, inplace = True)
geoData = gpd.read_file('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')
geoData = geoData.loc[~geoData['STATE'].isin(['02', '78','72','60','66','69','15']) ]
geoData  = geoData.drop(['GEO_ID', 'COUNTY', 'LSAD','geometry','STATE','CENSUSAREA'], axis=1)
geoData.rename(columns={'id':'FIPS','NAME': 'County'}, inplace=True)

clust = clust.merge(geoData,how = 'inner', on='FIPS')
clust['Moran I score for ACS billing fraction'] = clust['LISA_CL'].astype(str)
clust.loc[clust['Moran I score for ACS billing fraction'] == '0','Moran I score for ACS billing fraction'] = 'Non Significant'
clust.loc[clust['Moran I score for ACS billing fraction'] == '1','Moran I score for ACS billing fraction'] = 'High-High'
clust.loc[clust['Moran I score for ACS billing fraction'] == '2','Moran I score for ACS billing fraction'] = 'Low-Low'
clust.loc[clust['Moran I score for ACS billing fraction'] == '3','Moran I score for ACS billing fraction'] = 'Low-High'
clust.loc[clust['Moran I score for ACS billing fraction'] == '4','Moran I score for ACS billing fraction'] = 'High-Low'


clust.to_csv("redid complete morans I classified datalist.csv",index=False)


for x in['2019', '2018', '2017', '2016', '2015', '2015to2019']:
    clust[x] = 10000*gdf2[x]/gdf2['Beneficiaries with Part A and Part B']
clust['wit'] = 1
clust.drop(['FIPS'],axis=1, inplace = True)
df3 = anova_to_csv(pd.DataFrame(clust),clust.columns,'wit')
#make sure this indexes correctly
df3.to_csv("ent anova final2.csv",index=False)



