import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from matplotlib import rcParams

from glim_functions import plot_glim_no_basins, plot_glim_with_basins
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.
plt.rcParams['axes.labelweight'] = 'normal'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
rcParams['font.size'] = 14


chi_with_rain = False
dem_paths = ['./nevada/']#['/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/pyrenees/input_data/']
save_path = './nevada/'
location_name = ['nevada']
dem_files = ['nevada_COP30_UTM.bil']
hs_raster = ['nevada_COP30_UTM_hs.bil']
crs_number_cartopy = ['11']
south_h = False
count = 0

for location in location_name:
    crs = ccrs.UTM(zone=crs_number_cartopy[count],southern_hemisphere=south_h)
    map_ax = plt.subplot(projection=crs)
    #plot_glim_no_basins(map_ax, dem_paths[count], dem_files[count], location, save_path, hs_raster[count])
    plot_glim_with_basins(map_ax, dem_paths[count], dem_files[count], location, save_path, hs_raster[count])
    count+=1
