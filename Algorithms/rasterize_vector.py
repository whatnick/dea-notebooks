# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:24:26 2018
"""
import gdal

def rasterize_vector(input_data, cols, rows, geo_transform,
                     projection, field=None, raster_path=None):
    """
    Rasterize a vector file and return an array with values for cells that occur within the shapefile. 
    Can be used to obtain a binary array (shapefile vs no shapefile), or can assign the array cells with
    values from the shaepfile features by supplying the name of a shapefile field ('field). If 'raster_path' 
    is provided, the resulting array can be output as a geotiff raster.
    
    This function requires dimensions, projection data (in "WKT" format) and geotransform info 
    ("(upleft_x, x_size, x_rotation, upleft_y, y_rotation, y_size)") for the output array. 
    These are typically obtained from an existing raster using the following GDAL calls:
    
    # import gdal
    # gdal_dataset = gdal.Open(raster_path)
    # geotrans = gdal_dataset.GetGeoTransform()
    # prj = gdal_dataset.GetProjection()
    # out_array = gdal_dataset.GetRasterBand(1).ReadAsArray() 
    # yrows, xcols = out_array.shape
    
    Last modified: April 2018
    Author: Robbi Bishop-Taylor

    :attr input_data: input shapefile path or preloaded GDAL/OGR layer
    :attr cols: desired width of output array in columns. This can be obtained from an existing
                array using '.shape[0]')
    :attr rows: desired height of output array in rows. This can be obtained from an existing
                array using '.shape[1]')
    :attr geo_transform: geotransform for output raster; 
                 e.g. "(upleft_x, x_size, x_rotation, upleft_y, y_rotation, y_size)"
    :attr projection: projection for output raster (in "WKT" format)
    :attr field: shapefile field to rasterize values from. If none given (default), this 
                 assigns a value of 1 to all array cells within the shapefile, and 0 to 
                 areas outside the shapefile

    :returns: a 'row x col' array containing values from vector
    """

    # If input data is a string, import as shapefile layer
    if isinstance(input_data, str):
        # Open vector with gdal
        data_source = gdal.OpenEx(input_data, gdal.OF_VECTOR)
        input_data = data_source.GetLayer(0)

    # If raster path supplied, save rasterized file as a geotiff
    if raster_path:

        # Set up output raster
        print('Exporting raster to {}'.format(raster_path))
        driver = gdal.GetDriverByName('GTiff')
        target_ds = driver.Create(raster_path, cols, rows, 1, gdal.GDT_UInt16)

    else:

        # If no raster path, create raster as memory object
        driver = gdal.GetDriverByName('MEM')  # In memory dataset
        target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)

    # Set geotransform and projection
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)

    # Rasterize shapefile and extract array using field if supplied; else produce binary array
    if field:
        gdal.RasterizeLayer(target_ds, [1], input_data, options=["ATTRIBUTE=" + field])
    else:
        gdal.RasterizeLayer(target_ds, [1], input_data)    
    
    band = target_ds.GetRasterBand(1)
    out_array = band.ReadAsArray()
    target_ds = None

    return out_array