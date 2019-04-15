# Compare products from datacube. 

## Usage:  

- start VDI

- module use /g/data/v10/public/modules/modulefiles, if it is not automatically loaded when starting VDI.

- module load agdc-py3-prod

- run the compare_products_opensource.ipynb notebook 

  
## two parts of the comparing products

### Select products and load them from datacube

![Alt text](loading_data_gui.jpg?raw=true "Title")

- 1. Set up the database source by inputting the file name then pressing Enter

- 2. After the database source is set up, the available databases are shown in the Database choice drop down menu. Select one database. The 'default' database in Geoscience Australia is the operational database and can be used for comparing USGS level 2 data and GA ARD.

- 3. After selecting the database, all available products in the database are shown in the Product dropdown menu. Select one product, eg. ls8_ard.

- 4. After the product is selected, its available bands are shown in the Bands drop down menu. Select one or more bands for this product, eg. nbar_blue, nbar_green for ls8_ard.

- 5. Click the Add Product button

- 6. Repeat step 1 to 5, to add more products, eg. las8_usgs_l2c1

- 7. After adding all products, set start and end date, spatial location by select one of four choices: single window centering a location, multiple windows at multiple locations, single shapefile, or multiple shapefiles

- 8. Click Extract Products to load all selected products from datacube

- 9. Run the next cell in the Jupyter notebook, another GUI is produced for plotting.

![Alt text](ploting_data_gui.jpg?raw=true "Title")

- 10. Based on the products loaded earlier, in this GUI, the availble pairs of location and time are shown in the Location and time drop down menu. Select one of the location and time of your interest 

- 11. All products available for this pair of location and time are shown in the Products available drop down menu. Select the product of interest

- 12. After the product is selected, its previously loaded bands are shown for user to select one or more bands for plotting

- 13. Click the Add Products/Bands button to add the product with the selected bands for plotting.

- 14. Repeat step 10 to 13 to select all previously loaded products of interest for plotting 

- 15. After all products are selected, click the Plot button to plot these products.

![Alt text](ploting_data.jpg?raw=true "Title")

- 16. Then the time series data for one band for each product is drawn.

- 17. The data value range can be adjusted by typing the minimum value in the Min value box, and maximum value in the Max value box. 

- 18. From the drop down menu for each product user can select different band at one time to draw.

