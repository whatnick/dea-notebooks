# Compare products from datacube. 

## Set up environment and run:  

- start VDI

- module use /g/data/v10/public/modules/modulefiles, if it is not automatically loaded when starting VDI.

- module load dea

- note: for USGS users using AWS, the above processes can be skipped.

- run the compare_products_opensource.ipynb notebook 

  
## Two parts of the comparing products:

### Select products and load them from datacube

![Alt text](inter_comparison/docs/loading_data_gui.jpg?raw=true "Title")

- 1. Create a file containing the database sources. For example, create a file called database_sources.txt with content:
examples/.ard-interoperability.conf  
default  
while each line above points to a specific database. The 'default' database in Geoscience Australia is the operational database and can be used for comparing USGS level 2 data and GA ARD.

- 2. Set up the database source by inputting the file name then pressing Enter.

- 3. After the database source is set up, the available databases are shown in the Database choice drop down menu. Select one database e.g. examples/.ard-interoperability.conf. 

- 4. After selecting a database, all available products in the database are shown in the Product drop down menu. Select one product, e.g. ls8_ard. Note: loading all tables from the Geoscience Australia default operational database can take much longer time than expected, please be patient!

- 5. After the product is selected, its available bands are shown in the Bands to load drop down menu. Select one or more bands for this product, e.g. lambertian_coastal_aerosol.

- 6. In the Mask band drop down menu, select a band (e.g. fmask for ls8_ard) that is going to be used to mask cloud/shadow.

- 7. Click the Add Product button

- 8. Repeat step 1 to 7, to add more products, e.g. ls8_usgs_l2c1

- 9. After adding all products, set start and end date, spatial location by select one of four methods: single window centering a location, multiple windows at multiple locations, single polygon shapefile, or multiple polygon shapefiles.

- 10. Choose if adding the extra filters (only include valid observations, only include common dates between products to compare) by clicking the corresponding toggle buttons.

- 11. Click Extract Products to load selected bands for all selected products from datacube

- 12. Click Output Reports to export all bands for all selected products from datacube

- 13. To use the same parameters in the future, user can save the current setting for future use by typing a name in the Save Settings box at the bottom of the GUI. Then it will be available within the Load Settings box after the cell containing GUI is refreshed. Users can select the saved setting from the Load Settings box.

- 14. Run the next three cells in the Jupyter notebook, another GUI is produced for plotting.

![Alt text](inter_comparison/docs/ploting_data_gui.jpg?raw=true "Title")

- 15. Based on the products loaded earlier, in this GUI, the availble pairs of location and time are shown in the Location and time drop down menu. Select the location and time of your interest. 

- 16. All products available for this pair of location and time are shown in the Products available drop down menu. Select the product of interest

- 17. After the product is selected, its previously loaded bands are shown for user to select one or more bands for plotting

- 18. Click the Add Products/Bands button to add the product with the selected bands for plotting.

- 19. Repeat step 15 to 18 to select all previously loaded products of interest for plotting 

- 20. After all products are selected, click the Plot button to plot these products.

![Alt text](inter_comparison/docs/ploting_data.jpg?raw=true "Title")

- 21. Then the time series data for the band (chosen from all available bands) for each product is drawn.

- 22. The data value range can be adjusted by typing the minimum value in the Min value box, and maximum value in the Max value box. 

- 23. From the drop down menu for each product user can select different band at one time to draw.

