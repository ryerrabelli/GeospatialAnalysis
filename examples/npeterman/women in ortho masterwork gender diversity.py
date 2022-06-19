# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:35:15 2022

@author: nicho
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as stats

geoData = gpd.read_file('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')
geoData = geoData.loc[~geoData['STATE'].isin(['02', '78','72','60','66','69','15']) ]
geoData  = geoData.drop(['GEO_ID', 'COUNTY', 'NAME', 'LSAD'], axis=1)
statelist = geoData.STATE.unique()
geoData  = geoData.drop(['STATE'], axis=1)
geoData.rename(columns={'id':'FIPS'}, inplace=True)


gdf = pd.read_csv("ACS 2015 to 2019.csv")
gdf['FIPS'] = gdf['FIPS'].str[-5:]
gdf = pd.merge(geoData,gdf,how = 'inner', on='FIPS')
gdf.replace('-','0',inplace = True)
gdf.replace('N','0',inplace = True)
gdf['Population Density'] = gdf['Population']/gdf.CENSUSAREA
rural = pd.read_csv("Education.csv", usecols = ["fips","2013_Rural_urban_cont_code"])
rural = rural.dropna()
rural['fips'] = rural['fips'].astype(str).str.zfill(5)
rural.rename(columns={'fips':'FIPS'}, inplace=True)
gdf = gdf.merge(rural,how = 'inner', on='FIPS')

zip_county_cross = pd.read_csv("zip_county_cross.csv",usecols=["ZIP", "FIPS",'tot_ratio'], dtype=str)
zip_county_cross['FIPS'] = zip_county_cross['FIPS'].str.zfill(5) 
zip_county_cross['ZIP'] = zip_county_cross['ZIP'].str.zfill(5) 
zip_county_cross['tot_ratio'] = zip_county_cross['tot_ratio'].astype(float)
zip_county_cross = zip_county_cross.loc[zip_county_cross.groupby(['ZIP'])['tot_ratio'].idxmax()][['ZIP', 'FIPS']]

#read in all the column names, then use ifs to identify which columns you want
def readnpidata(file,zip_cross, numb):
    #good_col = ['NPI']
    npidata = pd.read_csv(file + '.csv',usecols = ['NPI','Entity Type Code','Provider Credential Text','Provider Business Practice Location Address Postal Code','Provider Enumeration Date','Provider Gender Code','Healthcare Provider Taxonomy Code_1','Healthcare Provider Taxonomy Code_2','Healthcare Provider Taxonomy Code_3'],dtype= "str")
    #cols = npidata.columns
    #for x in ['Entity Type Code','Provider Credential Text','Provider Business Practice Location Address Postal Code','Provider Enumeration Date','Provider Gender Code','Healthcare Provider Taxonomy Code_1','Healthcare Provider Taxonomy Code_2','Healthcare Provider Taxonomy Code_3']:
    #    if x in cols:
    #        good_col.append(x)
    npidata.rename(columns={'Provider Business Practice Location Address Postal Code':'ZIP','Healthcare Provider Taxonomy Code_1': 'Tax 1','Healthcare Provider Taxonomy Code_2': 'Tax 2','Healthcare Provider Taxonomy Code_3': 'Tax 3'}, inplace=True)
    npidata = npidata.loc[npidata['Entity Type Code'] == '1']
    npidata = npidata.loc[npidata['Provider Credential Text'].isin(['AG-ACNP','AGACNP-BC','AGNP-C','APN','APRN','AT-C, OT-C','ATC','C.R.N.P.','CERTIFIDE SURGICAL A','CNP','CPBMT,  OPA-C,  OTC','CRNP','CSA','CSFA','CST, SA','CST/CSFA','DNP, FNP - BC','D.P.M.','DNP, FNP-C','DPM','DPT','FNP','FNP PAC','FNP-BC','FNP-C','FRCS','FRCS(TR & ORTH), MSC','HEALTHCARE STUDENT','LPN','LPN, CCST','LPT','PAC','PH.D., LSA, OPA-C','PHARMD','PHYSICIAN ASSISTANT','PSC','STUDENT','SA','RST','RPA-C','RNFA','RNC','RN, RNFA','RN, MSN, APRN','RN','PTA','PT','PHYSICIANS ASST C','PHYSICIAN SURGEON FE','PA-C, R.T. (R)','PA-C, MMS','PA-C, MS','PA C','PA-C, MHP','PA-C','PA','P.T.','P.A.-C.','OT C','P.A.-C','P.A.,','P.A.','OTA','OT-C','OT','ORS','OPA-C RN','O.T.','NURSE PRACTITIONER','NP-C','NP','N.P.','MSN,RN, ACNP-BC','MSN, FNP','MS, PA-C','MS, ATC','MPH PA-C','MN']) == False]
    npidata.dropna(how= 'all',subset=['Tax 1','Tax 2','Tax 3','ZIP'],inplace = True)
    npidata.loc[npidata['Tax 1'].notnull(),'Tax 1'] = npidata.loc[npidata['Tax 1'].notnull(),'Tax 1'].str[:10]
    npidata.loc[npidata['Tax 2'].notnull(),'Tax 2'] = npidata.loc[npidata['Tax 2'].notnull(),'Tax 2'].str[:10]
    npidata.loc[npidata['Tax 3'].notnull(),'Tax 3'] = npidata.loc[npidata['Tax 3'].notnull(),'Tax 3'].str[:10]
    code = ['207X00000X']
    npidata =npidata.loc[(npidata['Tax 1'].notnull() & npidata['Tax 1'].isin(code))|(npidata['Tax 2'].notnull() & npidata['Tax 2'].isin(code)) | (npidata['Tax 3'].notnull() & npidata['Tax 3'].isin(code))]
    npidata['ZIP']= npidata['ZIP'].str[:5] 
    npidata['ZIP'] = npidata['ZIP'].astype(str).str.zfill(5)
    npidata['Total Orthopedic Surgeons' +numb] = 1
    npidata['Female Orthopedic Surgeons'+numb] = 0
    npidata.loc[npidata['Provider Gender Code'] == 'F','Female Orthopedic Surgeons'+numb] = 1
    npidata = npidata.groupby(['ZIP'], as_index=False).agg('sum') 
    npidata = npidata.merge(zip_cross ,how = 'inner', on='ZIP') 
    npidata = npidata.groupby(['FIPS'], as_index=False).agg('sum') 
    npidata['Gender Diversity Index'+numb] = 200*(1-((npidata['Female Orthopedic Surgeons'+numb]/npidata['Total Orthopedic Surgeons'+numb])**2 + (1-(npidata['Female Orthopedic Surgeons'+numb]/npidata['Total Orthopedic Surgeons'+numb]))**2))  
    return npidata

#append them all together then seperate

filelist = ['npidata20150607','npidata20151213','npidata20160410','npidata20161211','npidata20171112','npidata20180107','npidata_pfile20190113','npidata_pfile20190707','npidata_pfile20191211','npidata_pfile20200607','npidata_pfile20201213','npidata_pfile20210509','npidata_pfile20211107','npidata_pfile20220410']
masterfile = readnpidata(filelist[0],zip_county_cross,'0')
for x in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
    masterfile = masterfile.merge(readnpidata(filelist[x],zip_county_cross,str(x)),how = 'outer', on='FIPS') 
masterfile.fillna(0,inplace = True)
axisvalues= [6/12,12/12,16/12,24/12,35/12,37/12,49/12,55/12,60/12,66/12,72/12,77/12,83/12,88/12]

def calc_slope(row):
    a = stats.linregress(axisvalues, y=row)
    return a.slope 
def calc_rvalue(row):
    a = stats.linregress(axisvalues, y=row)
    return a.rvalue

masterfile['Gender Diversity Index: Slope'] = masterfile[['Gender Diversity Index0','Gender Diversity Index1','Gender Diversity Index2','Gender Diversity Index3','Gender Diversity Index4','Gender Diversity Index5','Gender Diversity Index6','Gender Diversity Index7','Gender Diversity Index8','Gender Diversity Index9','Gender Diversity Index10','Gender Diversity Index11','Gender Diversity Index12','Gender Diversity Index13']].apply(calc_slope,axis=1)
masterfile['Gender Diversity Index: Pearson Coef'] = masterfile[['Gender Diversity Index0','Gender Diversity Index1','Gender Diversity Index2','Gender Diversity Index3','Gender Diversity Index4','Gender Diversity Index5','Gender Diversity Index6','Gender Diversity Index7','Gender Diversity Index8','Gender Diversity Index9','Gender Diversity Index10','Gender Diversity Index11','Gender Diversity Index12','Gender Diversity Index13']].apply(calc_rvalue,axis=1)
masterfile['Gender Diversity Index: Average'] = masterfile[['Gender Diversity Index0','Gender Diversity Index1','Gender Diversity Index2','Gender Diversity Index3','Gender Diversity Index4','Gender Diversity Index5','Gender Diversity Index6','Gender Diversity Index7','Gender Diversity Index8','Gender Diversity Index9','Gender Diversity Index10','Gender Diversity Index11','Gender Diversity Index12','Gender Diversity Index13']].mean(axis=1)

masterfile['Female Orthopedic Surgeons: Slope'] = masterfile[['Female Orthopedic Surgeons0','Female Orthopedic Surgeons1','Female Orthopedic Surgeons2','Female Orthopedic Surgeons3','Female Orthopedic Surgeons4','Female Orthopedic Surgeons5','Female Orthopedic Surgeons6','Female Orthopedic Surgeons7','Female Orthopedic Surgeons8','Female Orthopedic Surgeons9','Female Orthopedic Surgeons10','Female Orthopedic Surgeons11','Female Orthopedic Surgeons12','Female Orthopedic Surgeons13']].apply(calc_slope,axis=1)
masterfile['Female Orthopedic Surgeons: Pearson Coef'] = masterfile[['Female Orthopedic Surgeons0','Female Orthopedic Surgeons1','Female Orthopedic Surgeons2','Female Orthopedic Surgeons3','Female Orthopedic Surgeons4','Female Orthopedic Surgeons5','Female Orthopedic Surgeons6','Female Orthopedic Surgeons7','Female Orthopedic Surgeons8','Female Orthopedic Surgeons9','Female Orthopedic Surgeons10','Female Orthopedic Surgeons11','Female Orthopedic Surgeons12','Female Orthopedic Surgeons13']].apply(calc_rvalue,axis=1)
masterfile['Female Orthopedic Surgeons: Average'] = masterfile[['Female Orthopedic Surgeons0','Female Orthopedic Surgeons1','Female Orthopedic Surgeons2','Female Orthopedic Surgeons3','Female Orthopedic Surgeons4','Female Orthopedic Surgeons5','Female Orthopedic Surgeons6','Female Orthopedic Surgeons7','Female Orthopedic Surgeons8','Female Orthopedic Surgeons9','Female Orthopedic Surgeons10','Female Orthopedic Surgeons11','Female Orthopedic Surgeons12','Female Orthopedic Surgeons13']].mean(axis=1)

masterfile["Total Orthopedic Surgeons: Slope"] = masterfile[['Total Orthopedic Surgeons0','Total Orthopedic Surgeons1','Total Orthopedic Surgeons2','Total Orthopedic Surgeons3','Total Orthopedic Surgeons4','Total Orthopedic Surgeons5','Total Orthopedic Surgeons6','Total Orthopedic Surgeons7','Total Orthopedic Surgeons8','Total Orthopedic Surgeons9','Total Orthopedic Surgeons10','Total Orthopedic Surgeons11','Total Orthopedic Surgeons12','Total Orthopedic Surgeons13']].apply(calc_slope,axis=1)
masterfile["Total Orthopedic Surgeons: Pearson Coef"] = masterfile[['Total Orthopedic Surgeons0','Total Orthopedic Surgeons1','Total Orthopedic Surgeons2','Total Orthopedic Surgeons3','Total Orthopedic Surgeons4','Total Orthopedic Surgeons5','Total Orthopedic Surgeons6','Total Orthopedic Surgeons7','Total Orthopedic Surgeons8','Total Orthopedic Surgeons9','Total Orthopedic Surgeons10','Total Orthopedic Surgeons11','Total Orthopedic Surgeons12','Total Orthopedic Surgeons13']].apply(calc_rvalue,axis=1)
masterfile["Total Orthopedic Surgeons: Average"] = masterfile[['Total Orthopedic Surgeons0','Total Orthopedic Surgeons1','Total Orthopedic Surgeons2','Total Orthopedic Surgeons3','Total Orthopedic Surgeons4','Total Orthopedic Surgeons5','Total Orthopedic Surgeons6','Total Orthopedic Surgeons7','Total Orthopedic Surgeons8','Total Orthopedic Surgeons9','Total Orthopedic Surgeons10','Total Orthopedic Surgeons11','Total Orthopedic Surgeons12','Total Orthopedic Surgeons13']].mean(axis=1)

#masterfile2 = masterfile.loc[masterfile["Total Orthopedic Surgeons: Average"] >= 1]
#axisvalues= [6,12,16,24,35,37,49,55,60,66,72,77,83,88]

gdf2 = pd.merge(gdf,masterfile,how = 'inner', on='FIPS')
gdf3 = gdf2.loc[gdf2["Total Orthopedic Surgeons: Average"] >= 1]
gdf3.to_file("orthowomen5.json", driver="GeoJSON")




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
clust = gpd.read_file('orthowomen_fixedanova.geojson')
clust.drop(['FIPS','CENSUSAREA','geometry'],axis=1, inplace = True)
clust[['% Families in Poverty: Single Mother ','Mean Travel Time to Work','% Households with Child < 6: Dual-Income','% Households with Child 6-17: Dual-Income','% Births: Unmarried']] = clust[['% Families in Poverty: Single Mother ','Mean Travel Time to Work','% Households with Child < 6: Dual-Income','% Households with Child 6-17: Dual-Income','% Births: Unmarried']].astype(float)
clust['wit'] = 1

df3 = anova_to_csv(pd.DataFrame(clust),clust.columns,'wit')


df3.to_csv("orthowomen final2.csv",index=False)














def calc_slope(row):
    a = stats.linregress(row, y=axisvalues)
    return pd.Series(a._asdict())
a1 = a1.set_index('FIPS')
z1 = a1.apply(calc_slope,axis=1)
z1 = a1.loc[:, a1.columns != 'FIPS'].apply(calc_slope,axis=1)








