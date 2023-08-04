from splot import esda as esdaplot
from splot.esda import plot_moran
from esda.moran import Moran
from libpysal.weights import Queen, Rook, KNN
import esda
from pysal.lib import weights

from numpy.random import seed
import geopandas as gpd
import pandas as pd
import pickle
import random
import pathlib
import json
import requests
import math
import pandana as pdna
import numpy as np
from scipy import stats
import itertools
import os
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
import functools
import tqdm
import shutil
from io import BytesIO
from zipfile import ZipFile

import housing_insecurity as housing_helper
from helper_functions import geospatial_helper

def download_file(query, filename):
      
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with urlopen(query) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(filename)
            
            

#####################################################
############## Download employment data #############
#####################################################
def download_LODES_data(state_abbrv):
    fname = state_abbrv+'_od_main_JT00_2019.csv.gz'
    filename = 'data/LODES/'+fname
    
    if pathlib.Path(filename).is_file():
        print('Employment Data for this State has been previously downloaded!')
        
    else:
        
        query = 'https://lehd.ces.census.gov/data/lodes/LODES7/'+state_abbrv+'/od/'+ fname
        download_file(query, filename)

    return filename

#####################################################
### Combine spatial, employment, and housing data ###
#####################################################
def process_LODES(od, gdf, housing, counties):
    od['w_county'] = od.w_geocode.str[2:5]
    od['h_county'] = od.h_geocode.str[2:5]
    od['w_census_tract'] = od.w_geocode.str[:11]
    od['h_census_tract'] = od.h_geocode.str[:11]
    od = od[(od.w_county.isin(counties)) & (od.h_county.isin(counties))]
    od = od.groupby(['h_census_tract', 'w_census_tract']).agg('sum').reset_index()

    h_gdf = gdf.rename({c:'h_'+c for c in gdf.columns}, axis=1)
    w_gdf = gdf.rename({c:'w_'+c for c in gdf.columns}, axis=1)
    od = od.merge(h_gdf, on='h_census_tract').merge(w_gdf, on='w_census_tract').sort_values(['h_census_tract', 
                                                                                                'w_census_tract'])
    od.reset_index(inplace=True, drop=True)
    
    od = od.merge(housing.rename({'census_tract':'h_census_tract',
                              'cluster_label':'h_housing'}, axis=1), how='left', on='h_census_tract')
    od = od.merge(housing.rename({'census_tract':'w_census_tract',
                              'cluster_label':'w_housing'}, axis=1), how='left', on='w_census_tract')
    return od


#####################################################
########### Global Spatial Autocorrelation ##########
#####################################################
def GSA(od, gdf, housing_grp, city_def, plot=False):
    total_commuting = od[['w_census_tract', 
                          'S000']].groupby('w_census_tract').sum().rename({'S000':'total'},
                                                                          axis=1).reset_index()
    housing_commuting = od[od.h_housing==housing_grp][['w_census_tract', 
                                                       'S000']].groupby('w_census_tract').sum().rename({'S000':housing_grp},
                                                                                                       axis=1).reset_index()
    commuting = total_commuting.merge(housing_commuting, how='left', on='w_census_tract')
    commuting['pct_employed'] = commuting[housing_grp]/commuting.total * 100
    commuting['pct_employed'] = commuting['pct_employed'].fillna(0)
    commuting = commuting.rename({'w_census_tract':'census_tract'},axis=1)
    commuting = gpd.GeoDataFrame(commuting.merge(gdf, how='left', on='census_tract'))
    
    morans_df = commuting.copy()
    w = Queen.from_dataframe(morans_df)
    w.transform = 'R'

    morans_df['pct_employed_lag']= weights.spatial_lag.lag_spatial(w, morans_df['pct_employed'])


    morans_df["pct_employed_std"] = morans_df["pct_employed"] - morans_df["pct_employed"].mean()
    morans_df["pct_employed_lag_std"] = (morans_df["pct_employed_lag"] - morans_df["pct_employed_lag"].mean())
    moran = esda.moran.Moran(morans_df["pct_employed"], w, permutations=1000)
    
    if plot:
        fig,ax=plt.subplots()
        sns.regplot(
            x="pct_employed_std",
            y="pct_employed_lag_std",
            ci=None,
            data=morans_df,
            color='grey',
            line_kws={"color": "r"},
            ax = ax
        )
        

        ax.set_title(city_def['city'].replace('_', ' ').title() + ': ' + str(round(moran.I,2)), fontsize=15)
        plt.savefig('figures/spatial autocorrelation/'+city_def['city']+'_global.pdf')
    
    
    return commuting, morans_df, moran.I, moran.p_sim

#####################################################
########### Local Spatial Autocorrelation ###########
#####################################################
def LSA(commuting, housing_grp, city_def):
    w = Queen.from_dataframe(commuting)
    w.transform = 'R'

    lag_field = housing_grp+'_lag'
    field_std = housing_grp+'_std'
    lag_field_std = lag_field+'_std'
    commuting[lag_field]= weights.spatial_lag.lag_spatial(w, commuting[housing_grp])
    commuting[field_std] = commuting[housing_grp] - commuting[housing_grp].mean()
    commuting[lag_field_std] = (commuting[lag_field] - commuting[lag_field].mean())
    lisa = esda.moran.Moran_Local(commuting[housing_grp], w)
    
    
    commuting['p-sim'] = lisa.p_sim
    sig = 1 * (lisa.p_sim < 0.05)
    commuting["sig"] = sig
    spots = lisa.q * sig
    spots_labels = {
        0: "Non-Significant",
        1: "HH",
        2: "LH",
        3: "LL",
        4: "HL",
    }

    commuting["labels"] = pd.Series(
        spots,
        index=commuting.index
    ).map(spots_labels)

    hotspot_tracts = list(commuting[commuting.labels=='HH'].census_tract)
    coldspot_tracts = list(commuting[commuting.labels=='CC'].census_tract)
    doughnut_tracts = list(commuting[commuting.labels=='LH'].census_tract)
    diamond_tracts = list(commuting[commuting.labels=='HL'].census_tract)
    
    return {'hotspot':hotspot_tracts,
            'coldspot':coldspot_tracts,
            'doughnut':doughnut_tracts,
            'diamond':diamond_tracts}
     
    
def visualise_hotspots(gdf, housing_GSA, housing_LSA, city):
    fig, axes = plt.subplots(2,3, figsize=(15,8))
    
    for i,hg in enumerate(['less', 'mild', 'most']):
        legend = False if i != 2 else True
        housing_GSA[hg][0].plot('pct_employed', vmin=0, vmax=100, ax=axes[0][i],
                                   cmap='Purples', legend=legend)
        

    for i,hg in enumerate(['less', 'mild', 'most']):
        gdf.plot(color='grey', ax=axes[1][i])

        hotspots =  housing_LSA[hg]['hotspot']
        gdf[gdf.census_tract.isin(hotspots)].plot(color=sns.color_palette('Reds',3)[i], 
                                                                        ax=axes[1][i])
        axes[1][i].set_title(hg.title()+' Vulnerable Hotspots')
    fig.suptitle(city.title())
    plt.savefig('figures/spatial autocorrelation/'+city+'_local.pdf')
    
def main(args):
    
    census_key = args[0]
    city_info_fp = args[1]
    
    housing_groups = ['mild', 'less', 'most']
    
    city,county_fips,state_fips,state_abbrv,state_full,city_def = housing_helper.get_city_info(city_info_fp)
    lodes_fpath = download_LODES_data(state_abbrv)
    od = pd.read_csv(lodes_fpath, dtype={'h_geocode':'str', 'w_geocode':'str'})
    gdf = geospatial_helper.process_state_shapefile(state_fips, county_fips)[['GEOID', 'INTPTLAT', 'INTPTLON', 'geometry']]
    gdf = gdf.rename({'GEOID':'census_tract'}, axis=1)
    housing = pd.read_csv('housing_clusters/'+city+'.csv', index_col=0, dtype={'census_tract':'str'})
    od = process_LODES(od, gdf, housing, county_fips)

    housing_GSA,housing_LSA={},{}
        
    for housing_grp in housing_groups:
        housing_GSA[housing_grp] = GSA(od, gdf, housing_grp, city_def, plot=True)
        housing_LSA[housing_grp] = LSA(housing_GSA[housing_grp][0], 'pct_employed', city_def)
        
        print(housing_grp, len(housing_LSA[housing_grp]['hotspot']))

    visualise_hotspots(gdf, housing_GSA, housing_LSA, city)
    
    return housing_GSA, housing_LSA
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
    
    