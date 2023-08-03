

import sys
import ast
import json
import requests
import shapely
import pandas as pd
import numpy as np
import geopandas as gpd


from sklearn.preprocessing import StandardScaler, normalize 

from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter


import sys
# print(sys.path)
# sys.path.append('helper_functions')

from helper_functions import ACS_helper
from helper_functions import geospatial_helper
from helper_functions import spectral_clustering

         


def get_city_info(city_info_fp):
    global city_info, state_full, state, state_fips, county_fips, city
    with open(city_info_fp) as f:
        city_info = dict(x.rstrip().split(':', 1) for x in f)
    for char in [' ', '"', "'", ',']:
        city_info = {k.strip(char):v.strip(char) for k,v in city_info.items()}
    city_info["counties"] = [county.zfill(3) for county in ast.literal_eval(city_info["counties"])]
    state_full = city_info["state"]
    state = city_info["state_abbrv"]
    state_fips = city_info["state_fips"].zfill(2)
    county_fips = city_info["counties"] 
    city = city_info["city"]
    return city,county_fips,state_fips,state,state_full,city_info


def process_evictionLab_data(evictions_df, gdf):
    evictions_df['date'] = pd.to_datetime(evictions_df['week_date'], format='%Y-%m-%d')
    evictions_df['year'] = [d.year for d in evictions_df.date]
    evictions_df['month'] = [d.month for d in evictions_df.date]

    evictions_df = evictions_df[(evictions_df.GEOID != 'sealed')]
    evictions_df = evictions_df[(evictions_df.year == 2020) & (evictions_df.month <= 6)]
    evictions_df = evictions_df.groupby('GEOID').sum().reset_index()[['filings_2020', 'GEOID']]
    evictions_df = evictions_df.rename({'GEOID':'census_tract', 'filings_2020':'evictions'},axis=1)
    return evictions_df

def get_evictions_sf():
    city = "san_francisco"
    state_full = 'california'
    state = 'ca'
    state_fips = '06'
    county_fips =['075']
    
#     water = geospatial_helper.download_water_shapefile(state_fips, county_fips)
    gdf = geospatial_helper.process_state_shapefile(state_fips, county_fips)
    
    evictions_df = pd.read_csv('/home/niyer/Ch3 Housing Insecurity/Housing Insecurity Data/rows.csv')

    evictions_df['date'] = pd.to_datetime(evictions_df['File Date'], format='%m/%d/%Y')
    evictions_df['year'] = [d.year for d in evictions_df.date]
    evictions_df['month'] = [d.month for d in evictions_df.date]
    evictions_df = evictions_df[evictions_df.year >= 2018]
    evictions_df = evictions_df[~evictions_df['Shape'].isna()]


    evictions_df['Longitude'] = [shapely.wkt.loads(row['Shape']).x for i,row in evictions_df.iterrows()]
    evictions_df['Latitude'] = [shapely.wkt.loads(row['Shape']).y for i,row in evictions_df.iterrows()]
    evictions_gdf = gpd.GeoDataFrame(evictions_df, 
                                         geometry=gpd.points_from_xy(evictions_df.Longitude, evictions_df.Latitude))
    pointInPoly = gpd.sjoin(evictions_gdf, gdf, how='left',op='intersects').merge(gdf, on='GEOID')


    evictions_df = pointInPoly.groupby('GEOID').agg({'geometry_y':'first', 'Eviction ID':'count'}).reset_index().rename({'GEOID':'census_tract','Eviction ID': 'evictions'}, axis=1)[['census_tract', 'evictions']]
    return evictions_df

def get_evictions_nyc():
    
    state_fips = '36'
    county_fips = ['005', '047', '061', '081', '085']
    gdf = geospatial_helper.process_state_shapefile(state_fips, county_fips)

    evictions_df = pd.read_csv('/home/niyer/Ch3 Housing Insecurity/Housing Insecurity Data/NYC_Evictions.csv')
    evictions_df['date'] = pd.to_datetime(evictions_df['Executed Date'], format='%m/%d/%Y')
    evictions_df['year'] = [d.year for d in evictions_df.date]
    evictions_df['month'] = [d.month for d in evictions_df.date]

    evictions_df = evictions_df[evictions_df.year >= 2018]
    evictions_df = evictions_df[(evictions_df.Longitude.notna()) & (evictions_df.Latitude.notna())]

    evictions_gdf = gpd.GeoDataFrame(evictions_df, 
                                         geometry=gpd.points_from_xy(evictions_df.Longitude, evictions_df.Latitude))

    pointInPoly = gpd.sjoin(evictions_gdf, gdf, how='left',op='intersects').merge(gdf, on='GEOID')

    evictions_df = pointInPoly.groupby('GEOID').agg({'geometry_y':'first', 'Court Index Number':'count'}).reset_index().rename({'GEOID':'census_tract', 'Court Index Number': 'evictions'}, axis=1)[['census_tract', 'evictions']]
    return evictions_df

def get_evictions(city, evictions_df=None):
    
    gdf = geospatial_helper.process_state_shapefile(state_fips, county_fips)
    global evictions
    if evictions_df is not None:
        if ('census_tract' not in evictions_df.columns) or ('evictions' not in evictions_df.columns):
            print("Eviction dataframe requires 'evictions' and 'census_tract' as its columns")
            return
    else:
        if city =='new_york_city':
            evictions_df = get_evictions_nyc()
        elif city == 'san_francisco':
            evictions_df = get_evictions_sf()
        else:
            evictions_df = pd.read_csv('data/eviction_lab_data.csv')
            evictions_df = evictions_df[evictions_df.city==city.replace('_',' ').title() + ', ' + state.upper()]
            evictions_df = process_evictionLab_data(evictions_df, gdf)
    evictions = evictions_df
    return evictions



def get_housing_clusters(housing_insecurity, std_housing_insecurity, city, 
                         plot_spectral=True, plot_cluster_distr=False):
    
    eigen_vecs, eigen_vals, spectral_gap = spectral_clustering.get_spectral_gap(std_housing_insecurity, city, 
                                                                                plot=plot_spectral)
    std_housing_insecurity_df = pd.DataFrame(std_housing_insecurity, columns=housing_insecurity.columns, 
                                             index=housing_insecurity.index)

    global clusters 
    clusters= []
    for i in range(1000):
        if i%50 == 0:
            print(i)
        kmeans_df, kmeans = spectral_clustering.spectral_kmeans(eigen_vecs, spectral_gap, std_housing_insecurity_df, 
                                                                plot=plot_cluster_distr)
        cluster_order = spectral_clustering.get_order_cluster(kmeans_df)
        housing_vuln = kmeans_df.reset_index().merge(cluster_order[['housing_cluster', 'cluster_label']], 
                                                     how='left', left_on='cluster_no',
                                                     right_on='housing_cluster').drop('housing_cluster',
                                                                                      axis=1)[['census_tract',
                                                                                               'cluster_label']]
        clusters.append(housing_vuln)

        
    clusters = pd.concat(clusters)
    clusters = clusters.groupby('census_tract').agg(lambda x: x.value_counts().index[0]).reset_index()
    clusters.to_csv('housing_clusters/'+city+'.csv')
      
        
def main(args):
    census_key = args[0]
    city_info_fp = args[1]
    evictions_df = args[2]
    get_city_info(city_info_fp)
    get_evictions(city, evictions_df=evictions_df)
    
    if (evictions is None) or (len(evictions) == 0):
        print('Evictions data missing')
        return
    
    acs5_variables_df = ACS_helper.get_acs_variables()
    housing_insecurity = ACS_helper.download_characteristics(state_fips, county_fips, acs5_variables_df, evictions, census_key)
    housing_insecurity, std_housing_insecurity = ACS_helper.preprocess_data(housing_insecurity, state_fips)

    
    get_housing_clusters(housing_insecurity, std_housing_insecurity, city, 
                         plot_spectral=True, plot_cluster_distr=False)
    
    print(city_info)
    
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
    
 