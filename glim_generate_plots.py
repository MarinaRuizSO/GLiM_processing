from osgeo import gdal
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import os
import cmcrameri.cm as cmc

import xarray as xr
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.


dem_names =  ['pyrenees_dem.bil']
location_name = ['pyrenees']
dem_path = ['/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/pyrenees/input_data/']
save_path = './'
dem_counter = 0 
for dem in dem_names:
    print(f'I am starting to process {dem}')
    #glim_file = "my_rasterized_column_glim_crs.tif"
    #glim_crs_dem = xr.open_dataset(glim_file, engine='rasterio')
    dem_raster = rasterio.open(dem_path[dem_counter] + dem_names[dem_counter])
    file_name = dem_path[dem_counter] + f'{location_name[dem_counter]}_dem.bil'
    clipper_name = f'{location_name[dem_counter]}_clipper.shp'
    output_name = f'{location_name[dem_counter]}_glim.tif'
    # os.system(f'gdaltindex {clipper_name} {file_path}+{file_name}')
    # os.system(f'gdalwarp -cutline {clipper_name} -crop_to_cutline {glim_file} {output_name}') 


    input_raster = gdal.Open(output_name)
    output_raster_name = f'{location_name[dem_counter]}_glim_crs_dem.tif'
    #breakpoint()
    warp = gdal.Warp(output_raster_name, input_raster, dstSRS=dem_raster.crs)
    warp = None # Closes the files

    # output_raster_new = rasterio.open(output_raster_name)

    # fig, ax = plt.subplots(figsize=(5, 5))

    # # use imshow so that we have something to map the colorbar to
    # # image_hidden = ax.imshow(np.squeeze(output_raster_new.read()), 
    # #                          cmap=cmc.hawaii)

    # # plot on the same axis with rio.plot.show
    # image =show(output_raster_new, 
    #         ax=ax, 
    #         cmap=cmc.hawaii)
    # ax.set_xlim(left=dem_raster.bounds[0], right=dem_raster.bounds[2])
    # plt.savefig(f'I_HATE_MAPS_SOMETIMES_TESTESTESTEST_{mountain}.png')
    dem_counter+=1