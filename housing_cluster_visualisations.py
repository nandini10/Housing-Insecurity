from helper_functions import geospatial_helper
from helper_functions import ACS_helper
import housing_insecurity as housing_helper


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


def housing_spatial_distr(insecurity_df, gdf, water_gdf, city, save=False, loc='upper right'):
    h,w = 25,60
    fig, (ax1) = plt.subplots(1,1, figsize=(w,h))
    cmap = plt.get_cmap('Reds', 5)

    cluster_gdf = gdf.merge(insecurity_df, how='left', left_on='GEOID', right_on='census_tract').fillna(0)
    cluster_gdf['cluster_label'] = ['NA' if row.cluster_label == 0 else row.cluster_label for i,row in cluster_gdf.iterrows()]
    cluster_gdf.plot('cluster_label', legend=True, figsize=(30,30), ax=ax1, edgecolor='black', cmap=cmap, linewidth=0,
                    legend_kwds={'prop':{'size':50}, 'markerscale':8, 'title':'Housing \n Vulnerability',
                                'title_fontsize':60,'labelspacing':0.8, 'loc':loc})#'handleheight':10, 
    
    plt.setp(ax1.get_legend().get_title(), multialignment='center')
    water_gdf.plot(color='#7393B3', ax=ax1, edgecolor='black')


    for ax in [ax1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set(xticklabels=['' for x in ax.get_xticklabels()])
        ax.set(yticklabels=['' for y in ax.get_yticklabels()])


    if save:
        
        plt.title('Spatial Distribution of Housing Clusters in ' + city.replace('_', ' ').title(), fontsize=70)
        
        plt.savefig('figures/housing geoviz/'+city+'_housing_spatial_distr.pdf',bbox_inches = "tight")

        
def get_housing_characteristics(insecurity_kmeans):
    # fig, axes = plt.subplots(len(nyc_features), 1, figsize=(10,70))
    features = [c for c in insecurity_kmeans.columns if (not 'cluster_label' in c)]
    groups = list(insecurity_kmeans.cluster_label.unique())
    kmeans_chars = pd.DataFrame(index=features, columns=groups)

    for feat in features:
        for g in groups:
            kmeans_chars[g][feat] = insecurity_kmeans[insecurity_kmeans.cluster_label==g][feat].mean()
            
    
    return kmeans_chars[['less','mild','most']], features
    
def plot_housing_characteristics(insecurity_kmeans, city, save=False):
    housing_labels_dict = {
    'rentBurdenPct': 'Severely Rent Burdened',
    'Mortgage': 'Mortgage',
    'Median gross rent': 'Median Gross Rent',
    'HousingPerPopulation': 'Housing Units by Population',
    'Lacking complete kitchen facilities:': '#Households w/o kitchen facilities',
    'PctLackingKitchen': 'Households w/o Kitchen',
    'PctLackingPlumbing': 'Households w/o Plumbing',
    'PctNoPhoneService': 'Households w/o Phone Service',
    'evictionsPerCapita': 'Evictions per Capita',
    'PPR': 'Persons per Rooms',
    'PPBR': 'Persons per Bedrooms',
}
    
    kmeans_chars, features = get_housing_characteristics(insecurity_kmeans)

    
    features_dict = {'Affordability': features[:3],
                     'Quality': features[3:6],
                     'Stability': features[6:]}
  
    
    fig, axes = plt.subplots(len(features_dict.keys()), 1, figsize=(30,30))
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set(xticklabels=['' for x in ax.get_xticklabels()])
    ax.set(yticklabels=['' for y in ax.get_yticklabels()])
    ax.set_xticks([])
    ax.set_yticks([])
    

    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    n_clusters=4
    cmap = plt.get_cmap('Greens', n_clusters)
    
    kmeans_chars.loc['HousingPerPopulation'] = kmeans_chars.loc['HousingPerPopulation']**-1 
    kmeans_chars.loc['Mortgage'] = kmeans_chars.loc['Mortgage']**-1 
    kmeans_chars_n = kmeans_chars.div(kmeans_chars.max(axis=1), axis=0).astype(float)
    for i,(k,v) in enumerate(features_dict.items()):
        kmean_chars_loc = kmeans_chars_n.loc[v]
        sns.heatmap(kmean_chars_loc, ax=axes[i],cbar=i == 0, cbar_ax=None if i else cbar_ax, 
                    cmap=plt.cm.Reds, linewidths=1, linecolor='black')
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=50)

        axes[i].set(yticklabels=[housing_labels_dict[feat] for feat in v])
        
        if i == 0:

            axes[i].set_title('Vulnerability of Clustered Groups', fontsize=70, pad=35)


        if i < len(features_dict)-1:
            axes[i].set(xticklabels=['' for y in axes[i].get_xticklabels()])
        else:
            axes[i].set_xticklabels([c.title() for c in kmeans_chars.columns], fontsize=50)

    cbar = axes[0].collections[0].colorbar
    kmeans_chars_values = kmeans_chars_n.loc[list(features_dict.values())[0]].values
    cbar.set_ticks([kmeans_chars_values.min(),
                    ((kmeans_chars_values.max()-kmeans_chars_values.min())/2)+kmeans_chars_values.min(),
                    kmeans_chars_values.max()])
    
    cbar.set_ticklabels(['low','middle', 'high'])
    
    cbar.ax.set_frame_on(True)
    cbar.ax.set_ylabel('Housing Insecurity', fontsize=60, rotation=270, labelpad=65)

    cbar.ax.tick_params(labelsize=50)
    plt.suptitle('Characteristics of Housing Demographics \n in ' + city.replace('_', ' ').title(),fontsize=90, y=1.15)
#     ax.set_xlabel('Vulnerability of Clustered Groups', fontsize=50, labelpad=50)

    
    if save:
        
        plt.savefig('figures/housing features/'+city+'_housing_features.pdf',bbox_inches = "tight")

        
        

def main(args):
    census_key = args[0]
    city_info_fp = args[1]
    evictions_df = args[2]
    save = args[3]
    city,county_fips,state_fips,state,state_full,city_info = housing_helper.get_city_info(city_info_fp)
    evictions_df = housing_helper.get_evictions(city, evictions_df=evictions_df)
    
    gdf = geospatial_helper.process_state_shapefile(state_fips, county_fips)
    water_gdf = geospatial_helper.download_water_shapefile(state_fips, county_fips)
    
    
    acs5_variables_df = ACS_helper.get_acs_variables()
    housing_insecurity = ACS_helper.download_characteristics(state_fips, county_fips, acs5_variables_df, evictions_df, census_key)
    housing_chars = housing_insecurity.copy()
    housing_insecurity, std_housing_insecurity = ACS_helper.preprocess_data(housing_insecurity, state_fips)

    
    housing_clusters = pd.read_csv('housing_clusters/'+city+'.csv', index_col=0, dtype={'census_tract':'str'})
    housing_insecurity = housing_insecurity.merge(housing_clusters, how='left', left_index=True,
                                                  right_on='census_tract')
    housing_insecurity= housing_insecurity.set_index('census_tract')
    
    if save :

        plot_housing_characteristics(housing_insecurity, city, save=True)
        housing_spatial_distr(housing_insecurity, gdf, water_gdf, city, save=True, loc='upper right')
        
    else:
        
        return housing_insecurity, housing_chars

    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])