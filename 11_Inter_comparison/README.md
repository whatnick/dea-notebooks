# Download USGS level 1 products using machine to machine method; order and download USGS level 2 products ( surface reflectance, temperature brightness, and pixel quality) using ESPA ordering sytem API. 

## Usage:  

- start VDI

- module use /g/data/v10/public/modules/modulefiles, if it is not automatically loaded when starting VDI.

- module load agdc-py3-prod

- download the files.py from the Support_files folder in order to create logs when running the script

- fill the content in the configuration files level1_download.cfg for Level 1 products and level2_order_download.cfg for Level 2 products 

- python level1_download.py -c level1_download.cfg  (for Level1 products, need to login as LPGS username to be able to run)

- python level2_order_download.py -c level2_order_download.cfg


# Compare the ARD products with the USGS level2 products. 

## compare_ard_usgs.ipynb notebook is used to compare the ard products and USGS level2 products. 

![Alt text](compare_gui.jpg?raw=true "Title")

## In the GUI, users are required to set following parameters:

- select one or multiple ard products (ls5_ard, ls7_ard, or ls8_ard)

- select one sub product (lambertian, nbar or nbart)

- set start and end date 

- spatial location: four choices: single window centering a location, multiple windows at multiple locations, single shapefile, or multiple shapefiles

- the output folder to hold the output csv files containing the attribute and statistical information for GA and USGS respectively
  

