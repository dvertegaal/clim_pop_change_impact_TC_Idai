#%% Use pixi environment compass-sfincs 
# This is an example of how the flood map resulting from the hydrodynamic modelling chain is masked
# Run the modelling chain from Vertegaal et al. (2026) with the correct CF values first and store the results in the data/sfincs directory
# You can find the modelling chain in the repository of Vertegaal et al. (2026) at https://zenodo.org/records/19067894

# Load modules
import os
from os.path import join
from hydromt_sfincs import SfincsModel, utils

#%% Set parameters
datacat              = [
    '../../data_catalogs/datacatalog_general.yml',
    '../../data_catalogs/datacatalog_SFINCS_coastal_coupling.yml',
    '../../data_catalogs/datacatalog_CF_forcing.yml'
    ]
CF_SLR_txt           = "-0.07"
CF_wind_txt          = "-4"
CF_rain_txt          = "-6"
model_name           = f"event_tp_era5_hourly_zarr_CF{CF_rain_txt}_GTSMv41_CF{CF_SLR_txt}_era5_hourly_spw_IBTrACS_CF{CF_wind_txt}"
dir_base             = f"../data/sfincs"
dir_run              = f"{dir_base}/{model_name}"
floodmap             = f"{dir_run}/plot_output/floodmap_15cm.tif"

#%%
print("------- Checking what we got ------")
print("Model run directory: ", dir_run)
print("Floodmap output: ", floodmap)

#%%
# select the model and datacatalog
sfincs_root = dir_run
mod = SfincsModel(sfincs_root, data_libs=datacat, mode="r")

# reading in the model results
mod.read_results()

#%%
# compute the maximum water level over all time steps
da_zsmax = mod.results["zsmax"].max(dim="timemax")

# select our highest-resolution elevation dataset
depfile = join(dir_base, "subgrid", "dep_subgrid.tif")
da_dep = mod.data_catalog.get_rasterdataset(depfile)

# we set a threshold to mask minimum flood depth above which people are affected
hmin = 0.15

# Downscale the floodmap
da_hmax = utils.downscale_floodmap(
    zsmax=da_zsmax,
    dep=da_dep,
    hmin=hmin,
    reproj_method = "bilinear",
)

# we use the GSWO dataset to mask permanent water by first reprojecting it to the subgrid of hmax
gswo = mod.data_catalog.get_rasterdataset("gswo", geom=mod.region, buffer=1000)
gswo_mask = gswo.raster.reproject_like(da_hmax, method="max")

# permanent water where water occurence > 5%
da_hmax_masked = da_hmax.where(gswo_mask <= 5)

# save the masked floodmap
da_hmax_masked.raster.to_raster(os.path.abspath(floodmap))

# %%
