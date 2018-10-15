# Download USGS level 2 products (surface reflectance, temperature brightness, and pixel quality) using ESPA ordering sytem API, prepare yaml files and ingest them into datacube. 

## download:  

- start VDI

- module use /g/data/v10/public/modules/modulefiles, if it is not automatically loaded when starting VDI.

- module load agdc-py3-prod

- fill the content in the configuration file level2_order_download.cfg for Level 2 products 

- python level2_order_download.py -c level2_order_download.cfg

## produce yaml files:

- ./unzip.sh to unzip the downloaded zip files 
- ./ledaps_lasrc_pbs_tasker.sh and ./do_qsub.sh to produce yaml files

## Ingest into datacube:

- datacube --config_file ~/.ard-interoperability.conf dataset add yamls/*/*.yaml

# Compare the ARD products with the USGS level2 products. 

## compare_ard_usgsl2.ipynb notebook is used to compare the ARD products and USGS level2 products. 

![Alt text](compare_gui.jpg?raw=true "Title")

## In the GUI, users are required to set following parameters:

- select one or multiple ard products (ls5_ard, ls7_ard, or ls8_ard)

- select one sub product (lambertian, nbar or nbart)

- set start and end date 

- spatial location: four choices: single window centering a location, multiple windows at multiple locations, single shapefile, or multiple shapefiles

- the output folder to hold the output csv files containing the attribute and statistical information for GA and USGS respectively
  

