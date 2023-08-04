# Housing-Insecurity
## Identifying US Census Tracts that are Vulnerable to Housing Insecurity

The code in this repository provide a framework for identifying regions in a city that are particularly vulnerable to housing insecurity. This code requires:

- A FIPS-based definition of a city, including
    * city name
    * state name, abbreviation, and FIPS code
    * FIPS code of all the counties included in the city
    
- Eviction Data in .csv format with at least the two following columns
    * `census_tract`: the 11-digit FIPS code referring to a census tract in the city
    * `evictions`: the number of evictions that happened in a census tract
        + we normalise this value with respect to the census tracts' population, so only raw counts are necessary
        
### ```Housing Vulnerability Demo.ipynb``` includes a tutorial of how to use the framework to (1) define clusters, (2) visualise results, (3) identify employment hotspots.
