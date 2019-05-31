# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:45:47 2019

@author: u89076
"""

from pyproj import Proj, transform
import rasterio
import rasterio.features
import numpy as np
from datacube.storage import masking
import warnings
from datacube import Datacube
import xarray as xr
import logging as log
from datetime import datetime
import plotly
import fiona
from datacube.utils import geometry
import os



# functions to retrieve data

def setQueryExtent(target_epsg, lon_cen, lat_cen, size_m):
    """
    Set the query extent in meteres as per the central geographical coordinates
    and the extent in meters.
    
    """
    
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init=target_epsg)    
    x_cen,y_cen = transform(inProj, outProj, lon_cen, lat_cen)
     
    x1 = x_cen - size_m /2 
    x2 = x_cen + size_m /2 
    y1 = y_cen + size_m /2 
    y2 = y_cen - size_m /2 
    
    return x1, y1, x2, y2    
    
    
def geometry_mask(geoms, geobox, all_touched=False, invert=False):
    """
    Create a mask from shapes.

    By default, mask is intended for use as a
    numpy mask, where pixels that overlap shapes are False.
    :param list[Geometry] geoms: geometries to be rasterized
    :param datacube.utils.GeoBox geobox:
    :param bool all_touched: If True, all pixels touched by geometries will be burned in. If
                             false, only pixels whose center is within the polygon or that
                             are selected by Bresenham's line algorithm will be burned in.
    :param bool invert: If True, mask will be True for pixels that overlap shapes.
    """
    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)
                                           
                                           
def get_pixel_size(dataset, source_band_list):
    """
    Decide the pixel size for loading the dataset from datacube
    """
    
    image_meta = dataset.metadata_doc['image']
    pixel_x_list = []
    pixel_y_list = []
    for a_band in source_band_list:
        if 'bands_info' in image_meta:
            # usgs level2
            pixel_x_list.append(int(image_meta['bands_info'][a_band]['pixel_size']['x']))
            pixel_y_list.append(abs(int(image_meta['bands_info'][a_band]['pixel_size']['y'])))
        else: 
            pixel_x_list.append(int(image_meta['bands'][a_band]['info']['geotransform'][1]))
            pixel_y_list.append(abs(int(image_meta['bands'][a_band]['info']['geotransform'][5])))

    pixel_x = min(pixel_x_list)
    pixel_y = min(pixel_y_list)
    
    return pixel_x, pixel_y  


def get_epsg(dataset):
    sr = dataset.metadata.grid_spatial['spatial_reference']
    if 'PROJCS' in sr:
        start_loc = sr.rfind('EPSG')
        epsg = '{}:{}'.format(sr[start_loc:start_loc+4], sr[start_loc+7:-3])
    else:
        # lansat ard
        epsg = sr
        
    return epsg


def remove_cloud_nodata(source_prod, data, mask_band): 
    ls8_USGS_cloud_pixel_qa_value = [324, 352, 368, 386, 388, 392, 400, 416, 432, 480, 
                                 864, 880, 898, 900, 904, 928, 944, 992, 1350]
    non_ls8_USGS_cloud_pixel_qa_value = [72, 96, 112, 130, 132, 136, 144, 160, 176, 224]
    non_ls8_USGS_sr_cloud_qa_value = [2, 4, 12, 20, 34, 36, 52]
    mask_data = data[mask_band]
    nodata_value = mask_data.nodata
    nodata_cloud_value = []
    
    if 'usgs' in source_prod:
        if 'ls8' in source_prod:
            nodata_cloud_value = ls8_USGS_cloud_pixel_qa_value
        else:
            if mask_band == 'sr_cloud_qa':
                nodata_cloud_value = non_ls8_USGS_sr_cloud_qa_value
            else:
                nodata_cloud_value = non_ls8_USGS_cloud_pixel_qa_value
                
        nodata_cloud_value.append(nodata_value)
        nodata_cloud = np.isin(mask_data, nodata_cloud_value) 
        cld_free = data.where(~nodata_cloud).dropna(dim='time', how='all')
    else:
        cld_free = data.where(mask_data == 1).dropna(dim='time', how='all')
           
    # remove nodata for the pixel of interest
    cld_free_valid = masking.mask_invalid_data(cld_free)
    
    return cld_free_valid

    
def cal_valid_data_per(data):
    print (data)
#     valid_pixel_per = np.count_nonzero(~np.isnan(band_info.loc[a_time].values)) * 100/ band_info.loc[a_time].values.size



def only_return_whole_scene(data):
    
    all_time_list = list(data.time.values)
    partial_time_list = []
    
    for band in data.data_vars:
        band_info = data[band]
        for a_time in all_time_list:
            valid_pixel_no = np.count_nonzero(~np.isnan(band_info.loc[a_time].values)) 
            # partial scenes
            if valid_pixel_no < band_info.loc[a_time].values.size:
                partial_time_list.append(a_time)
                break
    
    valid_time_list = list(set(all_time_list) - set(partial_time_list))
    valid_data = data.sel(time=valid_time_list).sortby('time')
    
    return valid_data
    
    
def round_time_ns(input_data):
    data = input_data
    for i in range(len(data.time)):
        data.time.values[i] = str(data.time.values[i])[:16]
    return data
    

def back2original_time_ns(data, orig_time):
    for i in range(len(data.time)):
        for a_time in orig_time:
            if str(data.time.values[i])[:16] in str(a_time):
                data.time.values[i] = a_time
                break            
    return data

    
def get_common_dates_data(items_list):
    # round time so to igonor difference on seconds
    original_time_list = []
    data_round_list = []
    for item in items_list:
        data_only = item[list(item.keys())[0]]['data']
        original_time_list.append(list(data_only.time.values))
        data_round_list.append(round_time_ns(data_only))

    # find common dates
    common_dates = set(data_round_list[0].time.values)
    for a_data in data_round_list[1:]:
        common_dates.intersection_update(set(a_data.time.values))
    
    # find the data with common dates and convert back to original time
    i = 0
    for a_data in data_round_list:
        data_common = a_data.sel(time=list(common_dates), method='nearest').sortby('time')
        data_common = back2original_time_ns(data_common, original_time_list[i])       
        # replace the old data with the common_original_data
        items_list[i][list(items_list[i].keys())[0]]['data'] = data_common
        i+=1
                
    return items_list    
    
    
def get_data_opensource(prod_info, input_lon, input_lat, acq_min, acq_max, 
                        window_size, no_partial_scenes): 
    
    datacube_config = prod_info[0]
    source_prod = prod_info[1]
    source_band_list = prod_info[2]
    mask_band = prod_info[3]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if datacube_config != 'default':
            remotedc = Datacube(config=datacube_config)
        else:
            remotedc = Datacube()

        return_data = {}
        data = xr.Dataset()

        if source_prod != '':
            # find dataset to get metadata
            fd_query = {        
                'time': (acq_min, acq_max),
                 'x' : (input_lon, input_lon+window_size/100000),
                 'y' : (input_lat, input_lat+window_size/100000),
                }
            sample_fd_ds = remotedc.find_datasets(product=source_prod, 
                                                  group_by='solar_day', **fd_query)                      

            if (len(sample_fd_ds)) > 0:
                # decidce pixel size for output data
                pixel_x, pixel_y = get_pixel_size(sample_fd_ds[0], source_band_list)
                log.info('Output pixel size for product {}: x={}, y={}'.format(source_prod, pixel_x, pixel_y))

                # get target epsg from metadata
                target_epsg = get_epsg(sample_fd_ds[0])
                log.info('CRS for product {}: {}'.format(source_prod, target_epsg))

                x1, y1, x2, y2 = setQueryExtent(target_epsg, input_lon, input_lat, window_size)

                query = {        
                    'time': (acq_min, acq_max),
                     'x' : (x1, x2),
                     'y' : (y1, y2),
                     'crs' : target_epsg,
                     'output_crs' : target_epsg,
                     'resolution': (-pixel_y, pixel_x),  
                     'measurements': source_band_list   
                    }

                if 's2' in source_prod:
                    data = remotedc.load(product=source_prod, 
                                         group_by='solar_day', **query)
                else:
                    data = remotedc.load(product=source_prod, 
                                         align=(pixel_x/2.0, pixel_y/2.0), 
                                         group_by='solar_day', **query)
                # remove cloud and nodta    
                data = remove_cloud_nodata(source_prod, data, mask_band)
                
                if no_partial_scenes:
                    # calculate valid data percentage
                    data = only_return_whole_scene(data)

            return_data = { 
                           source_prod: {'data': data, 'find_list': sample_fd_ds }
                          }
    
    return return_data 
    
    
# loading data by a shapefile
def get_data_opensource_shapefile(prod_info, acq_min, acq_max, shapefile, 
                                  no_partial_scenes):
    
    datacube_config = prod_info[0]
    source_prod = prod_info[1]
    source_band_list = prod_info[2]
    mask_band = prod_info[3]    
    
    if datacube_config != 'default':
        remotedc = Datacube(config=datacube_config)
    else:
        remotedc = Datacube()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fiona.open(shapefile) as shapes:
            crs = geometry.CRS(shapes.crs_wkt)
            first_geometry = next(iter(shapes))['geometry']
            geom = geometry.Geometry(first_geometry, crs=crs)            

            return_data = {} 
            data = xr.Dataset()
            
            if source_prod != '': 
                # get a sample dataset to decide the target epsg
                fd_query = {        
                    'time': (acq_min, acq_max),
                     'geopolygon': geom
                    }
                sample_fd_ds = remotedc.find_datasets(product=source_prod, 
                                                      group_by='solar_day',
                                                      **fd_query)

                if (len(sample_fd_ds)) > 0:
                    # decidce pixel size for output data
                    pixel_x, pixel_y = get_pixel_size(sample_fd_ds[0], source_band_list)
                    log.info('Output pixel size for product {}: x={}, y={}'.format(source_prod, pixel_x, pixel_y))

                    # get target epsg from metadata
                    target_epsg = get_epsg(sample_fd_ds[0])
                    log.info('CRS for product {}: {}'.format(source_prod, target_epsg))
                        
                    query = {
                            'time': (acq_min, acq_max),
                            'geopolygon': geom,
                            'output_crs' : target_epsg,
                            'resolution': (-pixel_y, pixel_x),
                            'measurements': source_band_list
                            }

                    if 's2' in source_prod:
                        data = remotedc.load(product=source_prod, 
                                             group_by='solar_day', **query)
                    else:
                        data = remotedc.load(product=source_prod, 
                                             align=(pixel_x/2.0, pixel_y/2.0), 
                                             group_by='solar_day', **query)
                    
                    # remove cloud and nodta    
                    data = remove_cloud_nodata(source_prod, data, mask_band) 
                    
                    if data.data_vars: 
                        mask = geometry_mask([geom], data.geobox, invert=True) 
                        data = data.where(mask)

                    if no_partial_scenes:
                        # calculate valid data percentage
                        data = only_return_whole_scene(data)                                             

                return_data = {
                               source_prod: {'data': data, 'find_list': sample_fd_ds }
                              }
                    
    return return_data 

    
    



def convert2original_loc_time(loc_time):
    if '.shp' in loc_time:
        orig_loc_time = (loc_time.split(' and ')[0], eval(loc_time.split(' and ')[1]))
    else:
        orig_loc_time = (eval(loc_time.split(' and ')[0]), eval(loc_time.split(' and ')[1]))
        
    return orig_loc_time



def draw_stat(plot_info, label, min_reflect, max_reflect, **kwargs): 
    
    plot_data_list = []  
    for key, value in plot_info.items():        
        data = value['data']
        colour = value['colour']
        
        plot_band = kwargs[key] 
        band_data = data[plot_band] 

        # mean value for all scenes over the polygon area
        mean = band_data.mean(dim=('x', 'y')).mean().values
        std = band_data.mean(dim=('x', 'y')).std().values
        
        # mean values for each scene over the polygon area
        mean_list = band_data.mean(dim=('x', 'y')).values 
        
        time_values_orig = band_data.time.values
        time_values = [datetime.strptime(str(d), '%Y-%m-%dT%H:%M:%S.%f000').date() for d in time_values_orig] 
        start_time = time_values[0]
        end_time = time_values[-1]

        plot_data = dict(        
            name = '{}_{}'.format(key, plot_band),
            x = time_values,
            y = mean_list,
            line = {
                'color': colour,
                'width': 1,}
            )

        plot_data_list.append(plot_data)

        plot_mean = dict(        
            name = '{}_{}_mean'.format(key, plot_band),
            x = [start_time, end_time],
            y = [mean, mean],
            line = {
                'color': colour,
                'width': 1,
                'dash': 'dash',}
            )
        plot_data_list.append(plot_mean)

        plot_std_pos = dict(        
            name = '{}_{}_std'.format(key, plot_band),
            x = [start_time, end_time],
            y = [mean+std, mean+std],
            line = {
                'color': colour,
                'width': 1,
                'dash': 'dashdot',}
            )
        plot_data_list.append(plot_std_pos)

        plot_std_nag = dict(        
            name = '{}_{}_std'.format(key, plot_band),
            x = [start_time, end_time],
            y = [mean-std, mean-std],
            line = {
                'color': colour,
                'width': 1,
                'dash': 'dashdot',}
            )
        plot_data_list.append(plot_std_nag)
    
    fig = dict(data=plot_data_list, layout={'title': str(label), 'yaxis': {'range': [min_reflect, max_reflect]}})
    plotly.offline.iplot(fig, filename='spectrum')
    
    
    
def creat_log_file():
    """
    
    """
    
    log_folder = os.path.join('.', 'logs')
    
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
           
    log_file = os.path.join(log_folder, '{}.log'.format(str(datetime.now())))
    
    return log_file