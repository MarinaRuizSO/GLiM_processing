import lsdviztools.lsdbasemaptools as bmt
from lsdviztools.lsdplottingtools import lsdmap_gdalio as gio
import lsdviztools.lsdmapwrappers as lsdmw


Dataset_prefix = "nevada"
source_name = "COP30"
DataDirectory = f"./{Dataset_prefix}/"


lower_left = [38.81445902714091,-114.95544433593749]
upper_right = [39.84270768759822, -114.24133300781249]

# YOU NEED TO PUT YOUR API KEY IN A FILE
your_OT_api_key_file = "my_OT_api_key.txt"

with open(your_OT_api_key_file, 'r') as file:
    print("I am reading you OT API key from the file "+your_OT_api_key_file)
    api_key = file.read().rstrip()
    print("Your api key starts with: "+api_key[0:4])



My_DEM = bmt.ot_scraper(source = source_name,
                        lower_left_coordinates = lower_left,
                        upper_right_coordinates = upper_right,
                        prefix = Dataset_prefix,
                        path = DataDirectory,
                        api_key_file = your_OT_api_key_file)
My_DEM.print_parameters()
My_DEM.download_pythonic()


Fname = Dataset_prefix+"_"+source_name+".tif"
gio.convert4lsdtt(DataDirectory,Fname)


# get the hillshade

## Get the basins
lsdtt_parameters = {"write_hillshade" : "true"}
r_prefix = Dataset_prefix+"_"+source_name +"_UTM"
w_prefix = Dataset_prefix+"_"+source_name +"_UTM"
lsdtt_drive = lsdmw.lsdtt_driver(command_line_tool = "lsdtt-basic-metrics",
                                 read_prefix = r_prefix,
                                 write_prefix= w_prefix,
                                 read_path = DataDirectory,
                                 write_path = DataDirectory,
                                 parameter_dictionary=lsdtt_parameters)
lsdtt_drive.print_parameters()
lsdtt_drive.run_lsdtt_command_line_tool()