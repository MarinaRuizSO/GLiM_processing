import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from matplotlib import rcParams

from glim_functions import plot_glim_no_basins
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.
plt.rcParams['axes.labelweight'] = 'normal'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
rcParams['font.size'] = 14


chi_with_rain = False
dem_paths = ['/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/pyrenees/input_data/']
save_path = './'
location_name = ['pyrenees']
dem_files = ['pyrenees_dem.bil']
hs_raster = ['pyrenees_dem_hs.bil']
crs_number_cartopy = ['31']
south_h = False
count = 0

for location in location_name:
    crs = ccrs.UTM(zone=crs_number_cartopy[count],southern_hemisphere=south_h)
    map_ax = plt.subplot(projection=crs)
    plot_glim_no_basins(map_ax, 'A', dem_paths[count], dem_files[count], location, save_path, hs_raster[count])
    count+=1
