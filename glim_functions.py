import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import cmcrameri.cm as cmc
import re
import figure_specs_paper

# Importing what I need, they are all installable from conda in case you miss one of them
import cartopy
import cartopy.mpl.geoaxes

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import lsdtopytools as lsd
# Ignore that last
# %load_ext xsimlab.ipython
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.
import cartopy.crs as ccrs
import rioxarray as rxr

#from functions_automate_chi_extraction import count_outlets_in_mountain, read_outlet_csv_file, get_dem_crs, coordinate_transform
import pyproj
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
#plt.rcParamaxes.labelweight
plt.rcParams['axes.labelweight'] = 'normal'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
rcParams['font.size'] = 14

def get_dem_crs(path, raster_name):
    my_dem = rasterio.open(path+raster_name)
    dem_crs = str(my_dem.crs)
    dem_crs = re.split('(\d+)',dem_crs)
    return int(dem_crs[1])

def plot_glim_data(mountain, DEM_path, DEM_file):
    glim_cropped_to_dem = rxr.open_rasterio(f'{mountain}_glim_crs_dem.tif', masked=True, decode_coords='all').squeeze()
    dem_mask = rasterio.open(DEM_path + DEM_file)

    glim_processed = glim_cropped_to_dem.fillna(99)

    # Plot data using nicer colors

    raster_no_nan = np.nan_to_num(np.squeeze(glim_processed), nan=99)
    unique_values = np.unique(raster_no_nan)

    # read colors from csv file
    litho_csv = pd.read_csv('./glim_lithology_codes.csv')
    
    # select only the legend labels that are in the raster
    unique_litho_csv = litho_csv[litho_csv['Number'].isin(unique_values)]

    # always add the nan values row
    colors = unique_litho_csv.Color.tolist()
    labels = unique_litho_csv.Lithology.tolist()
    # Create a list of labels to use for your legend


    class_bins = (np.array(unique_litho_csv.index.tolist())-0.5).tolist()+[100]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(class_bins,
                        len(colors))
    return norm, cmap, labels, dem_mask, glim_processed

def transform_coords_to_4326(dem_crs, x, y):
    proj = pyproj.Transformer.from_crs(dem_crs, 4326, always_xy=True)
    x1, y1 = (x, y)
    x2, y2 = proj.transform(x1, y1)
    print(f'lon, lat: {x2}, {y2}')
    return x2, y2


def add_basin_outlines(mydem, ax, size_outline = 1, zorder = 2, color = "k", alpha=0.5):
	"""
		Add catchment outlines to any axis given with its associated LSDDEM object ofc.
		No need to mention that catchment needs to be extracted within the LSDDEM object beforehand.
		Arguments:
			fig: the matplotlib figure
			ax: the matplotlib ax
			size_outline: size (in matplotlib units) of the outline
			zorder: the matplotlib zorder to use
			color (any color code compatible with matplotlib): the color of the outline. Default is black
		Returns:
			Nothing, It directly acts on the figure passed in argument
		Authors:
			B.G.
		Date:
			12/2018 (last update on the 23/02/2018)
	"""
	# getting dict of perimeters
	outlines = mydem.cppdem.get_catchment_perimeter()
	# keys are x,y,elevation,basin_key
	for key,val in outlines.items():

		ax.scatter(val["x"], val["y"], c = color, alpha = alpha, s = size_outline, zorder = zorder, lw =0)


# def plot_glim(axs, subplot_letter, mountain_count, DEM_paths, DEM_file, location_name, save_path):
#     print(f'I am processing {location_name[mountain_count]}')

#     lon_transform_list = []
#     lat_transform_list = []
#     filelist = count_outlets_in_mountain(DEM_paths[mountain_count])
#     count_file_list = 0
#     #crs = ccrs.UTM(crs_code)
#     #map_fig = plt.figure(figsize=(7,6))
#     #map_fig = figure_specs_paper.CreateFigure(FigSizeFormat="JGR", AspectRatio=16./9.)
#     #map_ax = plt.subplot(projection=crs)
#     #map_fig, map_ax = plt.subplots(1,1,figsize=(7,6), projection=crs)
#     cmap = cmc.hawaii#plt.cm.jet  # define the colormap
#     # extract all colors from the .jet map
#     cmaplist = [cmap(i) for i in range(cmap.N)]

#     # create the new map
#     cmap = mpl.colors.LinearSegmentedColormap.from_list(
#         'Custom cmap', cmaplist, cmap.N)
#     color_map = cmc.lajolla

#     dem_crs = get_dem_crs(DEM_paths[mountain_count],DEM_file)
#     dataset = rasterio.open(DEM_paths[mountain_count]+DEM_file)
#     dem_extent = dataset.bounds

#     lon_left, lat_bottom = transform_coords_to_4326(dem_crs, dem_extent[0], dem_extent[1])
#     # lon_right, lat_top = transform_coords_to_4326(dem_crs, dem_extent[2], dem_extent[3])
#     # output_raster_name = f'{mountain_range_names[mountain_count]}_glim_crs_dem.tif'
#     # output_raster_new = rasterio.open(output_raster_name)
    

#     for file in filelist[:]:
#         lat_outlet, lon_outlet = read_outlet_csv_file(file)
#         dem_crs = get_dem_crs(DEM_paths[mountain_count],DEM_file)
#         basin_name = file.split('/')[-1].split('__')[0]
#         lon_transform, lat_transform = coordinate_transform(lat_outlet, lon_outlet, dem_crs)
#         lon_transform_list.append(lon_transform)
#         lat_transform_list.append(lat_transform)
    
#         dem_basins_with_rain = lsd.LSDDEM(path = DEM_paths[mountain_count] ,file_name = DEM_file)
#         dem_basins_with_rain.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001)  


#         dem_basins_with_rain.ExtractRiverNetwork(method = "area_threshold", area_threshold_min = 200)


#         XY_basins_with_rain = dem_basins_with_rain.DefineCatchment(method="from_XY", test_edges = False, min_area = 0,max_area = 0, X_coords = lon_transform_list, Y_coords = lat_transform_list, 
#             coord_search_radius_nodes = 30, coord_threshold_stream_order = 3)
        
#         # load the dem information from the csv file

#         df_basins_with_rain_new = pd.read_csv(DEM_paths[mountain_count]+f'{basin_name}_no_rainfall__chi_map_theta_0_45.csv')
#         df_basins_with_rain = df_basins_with_rain_new[df_basins_with_rain_new.m_chi>=0]
        

#         #df_basins_with_rain = df_basins_with_rain[df_basins_with_rain.drainage_area>=0.01]
#         df_basins_with_rain.drainage_area/=np.max(df_basins_with_rain.drainage_area)
#         df_basins_with_rain = df_basins_with_rain[df_basins_with_rain.drainage_area>=0.01]
        
#         size_array = lsd.size_my_points(np.log10(df_basins_with_rain.drainage_area), 1,2)
#         axs.scatter(df_basins_with_rain.x, df_basins_with_rain.y, lw=0, c= "black",  zorder = 5, s=1)
#         #add_basin_outlines(dem_basins_with_rain, fig, axs, size_outline = 1, zorder = 4, color = "purple")

#         my_dem_raster = lsd.raster_loader.load_raster(DEM_paths[mountain_count]+DEM_file)

#         divider = make_axes_locatable(axs)
#         count_file_list+=1

    

#     print('I am about to go into the glim function')
#     norm, cmap, labels, dem_mask, glim_processed = plot_glim_data(location_name, DEM_paths )
#     print('I am going to plot the glim data')
#     im = glim_processed.plot.imshow(cmap=cmap,
#                                     norm=norm,
#                                     # Turn off colorbar
#                                     add_colorbar=False, ax = axs, alpha=0.75)
    

#     print('plotted! Now on to adding some extra shit')
#     # Add legend using earthpy
#     figure_specs_paper.draw_legend(im,titles=labels, size_font=10)
#     # im.axes.legend(
#     #     prop={"size": 13},
#     # )
#     #ax.legend(prop=dict(size=18))
#     #plt.setp(fontsize='xx-small')
#     #plt.legend(fontsize=10)
#     axs.set_xlim(left=dem_mask.bounds[0], right=dem_mask.bounds[2])
#     axs.get_xaxis().set_visible(True)
#     axs.get_yaxis().set_visible(True)
#     print('hi I will show plot maybe?')
#     latlon_df = pd.read_csv(DEM_paths+'map_locations_lookup_table.csv')
#     lonW = int(latlon_df[location_name][0])
#     lonE = int(latlon_df[location_name][1])
#     latS = int(latlon_df[location_name][2])
#     latN = int(latlon_df[location_name][3])
#     #-13.6927, -69.9541
#     axins = inset_axes(axs, width="30%", height="30%", loc=f'{latlon_df[location_name][4]}', 
#                     axes_class=cartopy.mpl.geoaxes.GeoAxes,
#                     axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
#     axins.add_feature(cartopy.feature.COASTLINE)
#     axins.add_feature(cartopy.feature.RIVERS)
#     axins.add_feature(cartopy.feature.LAKES)
#     axins.set_extent([lonW, lonE, latS, latN])
#     axins.stock_img()
#     axins.scatter([lon_left], [lat_bottom],
#         color='purple', linewidth=1, marker='*', zorder= 10, s = 40
        
#         )
    
#     # axs.text(-0.7, 0.9, subplot_letter, transform=axs.transAxes, 
#     #         weight='normal')
#     # plt.tight_layout()
#     # plt.show()
#     #src = rasterio.open(file_path+'/argentina/input_data/argentina_dem_hs.bil') 
#     topography = rxr.open_rasterio(DEM_paths+f'/{location_name}/input_data/{location_name}_dem_hs.bil', masked=True, decode_coords='all').squeeze()
#     #topography=rioxarray.open_rasterio(file_path+'/argentina/input_data/argentina_dem_hs.bil')
    
#     topography.plot.imshow(add_colorbar=False, ax = axs, alpha=0.25, cmap = cmc.grayC_r)
#     axs.set_title('')
#     # convert the x and y ticks to km 

#     axs.set_xticklabels((axs.get_xticks()/1000).astype(int))
#     axs.set_yticklabels((axs.get_yticks()/1000).astype(int))
#     axs.set_xlabel('Easting (km)')
#     axs.set_ylabel('Northing (km)')
#     axs.text(-0.3, 0.9, subplot_letter, transform=axs.transAxes, 
#             weight='normal')
#     plt.savefig(save_path+f'{location_name}_lithoGLim_all_basins.png', dpi=500, bbox_inches='tight')

def plot_glim_no_basins(axs, subplot_letter, DEM_path, DEM_file, location_name, save_path, hs_raster):
    print(f'I am processing {location_name}')


    cmap = cmc.hawaii#plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    dem_crs = get_dem_crs(DEM_path,DEM_file)
    dataset = rasterio.open(DEM_path+DEM_file)
    dem_extent = dataset.bounds

    lon_left, lat_bottom = transform_coords_to_4326(dem_crs, dem_extent[0], dem_extent[1])


    print('I am about to go into the glim function')
    norm, cmap, labels, dem_mask, glim_processed = plot_glim_data(location_name, DEM_path, DEM_file)
    print('I am going to plot the glim data')
    im = glim_processed.plot.imshow(cmap=cmap,
                                    norm=norm,
                                    # Turn off colorbar
                                    add_colorbar=False, ax = axs, alpha=0.75)
    

    print('plotted! Now on to adding some extra shit')
    # Add legend using earthpy
    figure_specs_paper.draw_legend(im,titles=labels, size_font=10)
    axs.set_xlim(left=dem_mask.bounds[0], right=dem_mask.bounds[2])
    axs.get_xaxis().set_visible(True)
    axs.get_yaxis().set_visible(True)
    print('hi I will show plot maybe?')
    latlon_df = pd.read_csv(save_path+'map_locations_lookup_table.csv')
    lonW = int(latlon_df[location_name][0])
    lonE = int(latlon_df[location_name][1])
    latS = int(latlon_df[location_name][2])
    latN = int(latlon_df[location_name][3])
    #-13.6927, -69.9541
    # axins = inset_axes(axs, width="30%", height="30%", loc=f'{latlon_df[location_name][4]}', 
    #                 axes_class=cartopy.mpl.geoaxes.GeoAxes,
    #                 axes_kwargs=dict(projection=ccrs.PlateCarree()))
    # axins.add_feature(cartopy.feature.COASTLINE)
    # axins.add_feature(cartopy.feature.RIVERS)
    # axins.add_feature(cartopy.feature.LAKES)
    # axins.set_extent([lonW, lonE, latS, latN])
    # axins.stock_img()
    # axins.scatter([lon_left], [lat_bottom],
    #     color='purple', linewidth=1, marker='*', zorder= 10, s = 40
        
    #     )
    
    topography = rxr.open_rasterio(DEM_path+hs_raster, masked=True, decode_coords='all').squeeze()
    
    topography.plot.imshow(add_colorbar=False, ax = axs, alpha=0.25, cmap = cmc.grayC_r)
    axs.set_title('')
    # convert the x and y ticks to km 

    axs.set_xticklabels((axs.get_xticks()/1000).astype(int))
    axs.set_yticklabels((axs.get_yticks()/1000).astype(int))
    axs.set_xlabel('Easting (km)')
    axs.set_ylabel('Northing (km)')

    plt.savefig(save_path+f'{location_name}_lithoGLim_all_basinsTEST.png', dpi=500, bbox_inches='tight')