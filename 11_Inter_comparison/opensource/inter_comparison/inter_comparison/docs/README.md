# Compare products from datacube. 

## Usage:  

- start VDI

- module use /g/data/v10/public/modules/modulefiles, if it is not automatically loaded when starting VDI.

- module load dea

- run the compare_products_opensource.ipynb notebook 

  
## two parts of the comparing products

### Select products and load them from datacube

 


- 1. Create a file containing the database sources. For example, create a file called database_sources.txt with content:
examples/.ard-interoperability.conf
default
while each line above points to a specific database. The 'default' database in Geoscience Australia is the operational database and can be used for comparing USGS level 2 data and GA ARD.

- 2. Set up the database source by inputting the file name then pressing Enter.

- 3. After the database source is set up, the available databases are shown in the Database choice drop down menu. Select one database e.g. default. 

- 4. After selecting a database, all available products in the database are shown in the Product drop down menu. Select one product, e.g. ls8_ard.

- 5. After the product is selected, its available bands are shown in the Bands to load drop down menu. Select one or more bands for this product, e.g. nbar_blue, nbar_green for ls8_ard.

- 6. In the Mask band drop down menu, select a band (e.g. fmask for ls8_ard) that is going to be used to mask cloud/shadow.

- 7. Click the Add Product button

- 8. Repeat step 1 to 7, to add more products, e.g. ls8_usgs_l2c1

- 9. After adding all products, set start and end date, spatial location by select one of four methods: single window centering a location, multiple windows at multiple locations, single polygon shapefile, or multiple polygon shapefiles.

- 10. Choose if adding the extra filters (only include valid observations, only include common dates between products to compare) by clicking the corresponding toggle buttons.

- 11. Click Extract Products to load all selected products from datacube

- 12. Run the next three cells in the Jupyter notebook, another GUI is produced for plotting.

 


- 13. Based on the products loaded earlier, in this GUI, the availble pairs of location and time are shown in the Location and time drop down menu. Select the location and time of your interest. 

- 14. All products available for this pair of location and time are shown in the Products available drop down menu. Select the product of interest

- 15. After the product is selected, its previously loaded bands are shown for user to select one or more bands for plotting

- 16. Click the Add Products/Bands button to add the product with the selected bands for plotting.

- 17. Repeat step 13 to 16 to select all previously loaded products of interest for plotting 

- 18. After all products are selected, click the Plot button to plot these products.

 


- 19. Then the time series data for the band (chosen from all available bands) for each product is drawn.

- 20. The data value range can be adjusted by typing the minimum value in the Min value box, and maximum value in the Max value box. 

- 21. From the drop down menu for each product user can select different band at one time to draw.

