#%% Use pixi environment compass-socio to run this script
import os
from os.path import join
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import rioxarray as rxr
import rasterio
from rasterio.mask import mask
from rasterio import features
from rasterio.transform import array_bounds
from shapely.geometry import box, mapping, Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, mcolors
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


# Get extent from raster transform
def get_extent(transform, width, height):
    left = transform[2]
    right = left + width * transform[0]
    top = transform[5]
    bottom = top + height * transform[4]
    return [left, right, bottom, top]

# # ===== CONFIGURATION =====
# EVENT_NAME = "Idai"
BASE_DATA_PATH = Path("../data")
SCENARIO_PATH_F = "event_tp_era5_hourly_zarr_CF0_GTSMv41_CF0_era5_hourly_spw_IBTrACS_CF0" # factual


# ===== FILE PATHS =====
# Factual and counterfactual flood maps (from sfincs)
F_flooding = BASE_DATA_PATH / "sfincs" / SCENARIO_PATH_F / "floodmap_15cm.tif"

# population raster from the Global Human Settlement Layer (GHSL) database (original 100 m resolution)
# You can download the GHSL population rasters for 2020 and 1975 here: https://human-settlement.emergency.copernicus.eu/download.php?ds=pop
pop_raster_path_GHSL_100m_2020 = Path(BASE_DATA_PATH / "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R12_C22.tif")
pop_raster_path_GHSL_100m_2015 = Path(BASE_DATA_PATH / "GHS_POP_E2015_GLOBE_R2023A_54009_100_V1_0_R12_C22.tif")
pop_raster_path_GHSL_100m_1975 = Path(BASE_DATA_PATH / "GHS_POP_E1975_GLOBE_R2023A_54009_100_V1_0_R12_C22.tif")

# Population raster from GLOPOP-SG - preprocessed by Doris Vertegaal
pop_df_path_GLOPOP_SG = Path("../data/GLOPOP-SG/synthpop_MOZr107_grid_combined.csv")  # synthesized population data with coordinates
df_pop_charac = pd.read_csv(pop_df_path_GLOPOP_SG)
pop_path_GLOPOP_SG_1km = Path(BASE_DATA_PATH / "GLOPOP-SG" / "MOZr107_population.tif")  # synthesized population data rasterized to 1km grid
pop_path_GLOPOP_SG_25m = Path(BASE_DATA_PATH / "GLOPOP-SG" / "MOZr107_population_regrid.tif") # synthesized population data regridded to flood grid (25m) from Socio-economic/scripts/analyse_synthpop_data_GLOPOP-SG.py

#%%
# --- Read flood rasters ---
hmax_F_da = rxr.open_rasterio(F_flooding).squeeze("band", drop=True) 
hmax_F = hmax_F_da.values

# Flood model subgrid
sfincs_subgrid = os.path.join(BASE_DATA_PATH, "sfincs", SCENARIO_PATH_F, "subgrid", "dep_subgrid.tif")

# --- Flood grid properties ---
with rasterio.open(sfincs_subgrid) as src:
    flood_grid_crs, flood_grid_transform, flood_grid_shape = src.crs, src.transform, (src.height, src.width)

height, width = hmax_F.shape
left, bottom, right, top = array_bounds(height, width, flood_grid_transform)
flood_extent = [left, right, bottom, top]

#%%
# ==== GEOMETRIES AND MASKS ====
# Load study region
region = gpd.read_file(BASE_DATA_PATH / "sfincs" / SCENARIO_PATH_F / "gis" / "region.geojson")
region_wgs84 = region.to_crs("EPSG:4326")
region_utm = region.to_crs(flood_grid_crs)
region_geom_poly = region_wgs84.geometry.iloc[0]

# Rasterio-compatible geometries
region_geom = [region_wgs84.geometry.iloc[0].__geo_interface__]

with rasterio.open(pop_raster_path_GHSL_100m_2020) as src:
    region_mw = region.to_crs(src.crs)
region_geom_mw = [region_mw.geometry.iloc[0].__geo_interface__]

# =============================================================================
# Administrative boundaries
districts_adm3 = gpd.read_file(BASE_DATA_PATH / "gis" / "sofala_districts_study_region.shp")
districts_adm3_utm = (districts_adm3.to_crs(region.crs).overlay(region, how="intersection"))

districts_adm2 = (gpd.read_file(BASE_DATA_PATH / "gis" / "gadm41_MOZ_2.shp").to_crs(region_wgs84.crs))
districts_adm2 = districts_adm2.loc[districts_adm2.intersects(region_geom_poly).copy()]
districts_adm2_utm = districts_adm2.to_crs(region.crs)

# Remove districts outside study region
districts_adm3_filtered = districts_adm3_utm.loc[~districts_adm3_utm["NAME_3"].isin(["Muanza", "Gororngosa-Sede", "Galinha"])]
districts_adm2_filtered = districts_adm2_utm.loc[~districts_adm2_utm["NAME_2"].isin(["Muanza", "Gorongosa"])]

# =============================================================================
# Background layers
background = (gpd.read_file(BASE_DATA_PATH / "gis" / "case_study_region_background.geojson").to_crs("EPSG:4326"))
background_utm = background.to_crs("EPSG:32736")  # UTM Zone 36S

shapefile_sofala = gpd.read_file(BASE_DATA_PATH / "gis" / "sofala_province.shp")


with rasterio.open(pop_raster_path_GHSL_100m_2020) as src_GHSL_100m_2020:
    pop_GHSL_100m_2020, transform_GHSL_100m_2020 = mask(src_GHSL_100m_2020, region_geom_mw, crop=True)
    print("CRS GHSL 2020:", src_GHSL_100m_2020.crs)  
    print("No-data value GHSL 2020:", src_GHSL_100m_2020.nodata)
    pop_GHSL_100m_2020 = np.where(pop_GHSL_100m_2020 == -200, np.nan, pop_GHSL_100m_2020)
    print("Remaining -200 values:", np.sum(pop_GHSL_100m_2020 == -200))

with rasterio.open(pop_raster_path_GHSL_100m_2015) as src_GHSL_100m_2015:
    pop_GHSL_100m_2015, transform_GHSL_100m_2015 = mask(src_GHSL_100m_2015, region_geom_mw, crop=True)
    print("No-data value GHSL 2015:", src_GHSL_100m_2015.nodata)
    pop_GHSL_100m_2015 = np.where(pop_GHSL_100m_2015 == -200, np.nan, pop_GHSL_100m_2015)
    print("Remaining -200 values:", np.sum(pop_GHSL_100m_2015 == -200))

with rasterio.open(pop_raster_path_GHSL_100m_1975) as src_GHSL_100m_1975:
    pop_GHSL_100m_1975, transform_GHSL_100m_1975 = mask(src_GHSL_100m_1975, region_geom_mw, crop=True)
    print("No-data value GHSL 1975:", src_GHSL_100m_1975.nodata)
    pop_GHSL_100m_1975 = np.where(pop_GHSL_100m_1975 == -200, np.nan, pop_GHSL_100m_1975)
    print("Remaining -200 values:", np.sum(pop_GHSL_100m_1975 == -200))

extent_GHSL_2020 = get_extent(transform_GHSL_100m_2020, pop_GHSL_100m_2020.shape[2], pop_GHSL_100m_2020.shape[1])
extent_GHSL_2015 = get_extent(transform_GHSL_100m_2015, pop_GHSL_100m_2015.shape[2], pop_GHSL_100m_2015.shape[1])
extent_GHSL_1975 = get_extent(transform_GHSL_100m_1975, pop_GHSL_100m_1975.shape[2], pop_GHSL_100m_1975.shape[1])


#%%
################################################################################
########################### COMPARING TOTAL POPUATION ##########################
################################################################################
print("Total population in study region")
print(f"pop_GHSL_100m_2020: {np.nansum(pop_GHSL_100m_2020):,.0f}")
print(f"pop_GHSL_100m_2015: {np.nansum(pop_GHSL_100m_2015):,.0f}")
print(f"pop_GHSL_100m_1975: {np.nansum(pop_GHSL_100m_1975):,.0f}")


#%%
################################################################################
########################## COMPARING EXPOSED POPUATION #########################
################################################################################
############### REGRIDDING POPULATION TO FLOOD GRID ###############
# --- Function to redistribute population over land pixels on flood grid ---
def reproject_and_redistribute_population_over_land(pop_path, land_gdf, flood_crs, flood_transform, flood_shape, 
                                                    province_geom=None, region=None, districts_adm3=None, districts_adm2=None,
                                                    year=None, out_raster_path=None, source=None):

    print(f"▶ Loading {year} population raster...")

    if out_raster_path is not None and os.path.exists(out_raster_path):
        print(f"▶ Loading existing raster from {out_raster_path}")
        with rasterio.open(out_raster_path) as src:
            pop_fine = src.read(1)
            return pop_fine
    # ------------------------------------------------------------
    # 1. READ POPULATION (KEEP NATIVE CRS)
    # ------------------------------------------------------------
    with rasterio.open(pop_path) as src:
        pop_crs = src.crs
        region_native = region.to_crs(pop_crs)
        region_geom = [mapping(geom) for geom in region_native.geometry]

        pop, pop_affine = rasterio.mask.mask(
            src,
            region_geom,
            crop=True
        )

    pop = pop[0].astype("float32")

    nodata = src.nodata
    if nodata is not None:
        pop[pop == nodata] = np.nan

    pop[pop <= 0] = np.nan         

    # ------------------------------------------------------------
    # 2. OUTPUT GRID
    # ------------------------------------------------------------
    pop_fine = np.zeros(flood_shape, dtype=np.float32)

    # Land mask (already in flood CRS)
    land_mask = features.rasterize(
        [(geom, 1) for geom in land_gdf.geometry],
        out_shape=flood_shape,
        transform=flood_transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    print("  ✔ Land mask created on flood grid.")

    # ------------------------------------------------------------
    # 3. REDISTRIBUTION LOOP
    # ------------------------------------------------------------
    print("▶ Redistributing population to fine grid...")

    for row in tqdm(range(pop.shape[0]), desc="Processing coarse cells"):
        for col in range(pop.shape[1]):

            pop_value = pop[row, col]

            if np.isnan(pop_value) or pop_value <= 0:
                continue

            # ----------------------------------------------------
            # Pixel bounds in ORIGINAL CRS
            # ----------------------------------------------------
            x_min, y_max = pop_affine * (col, row)
            x_max, y_min = pop_affine * (col + 1, row + 1)

            coarse_cell = box(x_min, y_min, x_max, y_max)

            # ----------------------------------------------------
            # Reproject cell to FLOOD CRS
            # ----------------------------------------------------
            coarse_cell_flood = gpd.GeoSeries(
                [coarse_cell],
                crs=pop_crs
            ).to_crs(flood_crs).iloc[0]

            # ----------------------------------------------------
            # Rasterize onto flood grid
            # ----------------------------------------------------
            coarse_mask = features.rasterize(
                [(mapping(coarse_cell_flood), 1)],
                out_shape=flood_shape,
                transform=flood_transform,
                fill=0,
                dtype=np.uint8
            ).astype(bool)

            valid_mask = coarse_mask & land_mask
            n_valid = valid_mask.sum()

            if n_valid == 0:
                continue

            # ----------------------------------------------------
            # Integer redistribution
            # ----------------------------------------------------
            P = int(round(pop_value))
            base = P // n_valid
            remainder = P % n_valid

            idx = np.where(valid_mask)

            pop_fine[idx] += base

            if remainder > 0:
                perm = np.random.permutation(len(idx[0]))
                chosen = perm[:remainder]
                pop_fine[idx[0][chosen], idx[1][chosen]] += 1

    # ------------------------------------------------------------
    # 4. VALIDATION
    # ------------------------------------------------------------
    total_input = float(np.nansum(pop))
    total_output = float(pop_fine.sum())

    diff = abs(total_output - total_input)
    rel_diff = diff / total_input * 100

    print("  ✔ Redistribution done.")
    print(f"  🔹 Input population:  {total_input:,.0f}")
    print(f"  🔹 Output population: {total_output:,.0f}")
    print(f"  🔹 Difference:        {diff:,.2f} ({rel_diff:.6f} %)")

    if rel_diff > 0.01:
        print("  ⚠ WARNING: population not fully conserved")

    # ------------------------------------------------------------
    # 5. SAVE OUTPUT
    # ------------------------------------------------------------
    if out_raster_path is not None:

        H, W = pop_fine.shape
        a, b, c, d, e, f = flood_transform[:6]

        profile = {
            "driver": "GTiff",
            "dtype": rasterio.float32,
            "count": 1,
            "height": H,
            "width": W,
            "crs": flood_crs,
            "transform": flood_transform,
            "compress": "deflate"
        }

        with rasterio.open(out_raster_path, "w", **profile) as dst:
            dst.write(pop_fine, 1)

        print(f"▶ Saved: {out_raster_path}")

    return pop_fine

# --- Function to link population raster to flood depth as DataFrame ---
def pop_raster_to_gdf(pop_array, flood_array, settlement_type_array, transform, year, climate, export_df=True, export_path=None):
    print("Linking population raster to flood depth as DataFrame...")

    # Check if shapes match
    if pop_array.shape == flood_array.shape == settlement_type_array.shape:
        print("✔ Shapes match")
    else:
        print("✖ Shapes do NOT match!", pop_array.shape, flood_array.shape, settlement_type_array.shape)

    # Flatten arrays
    pop_flat = pop_array.ravel()
    flood_flat = flood_array.ravel()
    settlement_type_flat = settlement_type_array.ravel()

    # Mask zero-pop cells
    mask = (pop_flat > 0) & (flood_flat > 0)
    pop_vals = pop_flat[mask]
    flood_vals = flood_flat[mask]
    settlement_type_vals = settlement_type_flat[mask]

    # Pixel coordinates (centers)
    rows, cols = np.indices(pop_array.shape)
    xs, ys = transform * (cols.ravel()[mask] + 0.5, rows.ravel()[mask] + 0.5)

    df = pd.DataFrame({
        "population": pop_vals,
        "flood_depth": flood_vals,
        "settlement_type": settlement_type_vals,
        "x": xs,
        "y": ys
    })

    if export_df:
        file_name = f"df_pop_{year}_{climate}.csv"
        df.to_csv(join(export_path, file_name), index=False)
        print(f"▶ Exported DataFrame to {join(export_path, file_name)}")

    return df

# --- Function to aggregate population and flood depth to coarser grid ---
def aggregate_pop(total_pop_array, flood_raster, transform, crs, region=None, background=None, factor=100):
    print("Aggregating population raster to coarser grid polygons...")

    # Pixel size from transform
    pixel_width = transform.a
    pixel_height = -transform.e

    # Block (coarse) size in meters
    cell_width = factor * pixel_width
    cell_height = factor * pixel_height

    # --- Helper functions ---
    def block_sum(arr, factor):
        nrows, ncols = arr.shape
        nrows_crop = nrows - nrows % factor
        ncols_crop = ncols - ncols % factor
        arr_cropped = arr[:nrows_crop, :ncols_crop]
        return arr_cropped.reshape(nrows_crop//factor, factor, ncols_crop//factor, factor).sum(axis=(1,3))

    def block_mean(arr, factor):
        nrows, ncols = arr.shape
        nrows_crop = nrows - nrows % factor
        ncols_crop = ncols - ncols % factor
        arr_cropped = arr[:nrows_crop, :ncols_crop]
        return np.nanmean(arr_cropped.reshape(nrows_crop//factor, factor, ncols_crop//factor, factor), axis=(1,3))

    def block_count_threshold(arr, factor, threshold=1):
        nrows, ncols = arr.shape
        nrows_crop = nrows - nrows % factor
        ncols_crop = ncols - ncols % factor
        arr_cropped = arr[:nrows_crop, :ncols_crop]
        return (arr_cropped.reshape(nrows_crop//factor, factor, ncols_crop//factor, factor) > threshold).sum(axis=(1,3))

    # --- Aggregate population and flood ---
    total_agg = block_sum(total_pop_array, factor)
    exposed_agg = block_sum(np.where(flood_raster > 0, total_pop_array, 0), factor)
    avg_flood_depth = block_mean(flood_raster, factor)
    cells_flooded = block_count_threshold(flood_raster, factor=factor, threshold=0)
    pixels_high = block_count_threshold(flood_raster, factor=factor, threshold=1)
    pixels_higher = block_count_threshold(flood_raster, factor=factor, threshold=1.5)

    # --- Build coarse grid ---
    nrows_coarse, ncols_coarse = total_agg.shape
    x0, y0 = transform * (0, 0)
    x_coords = x0 + np.arange(ncols_coarse) * cell_width
    y_coords = y0 + np.arange(nrows_coarse) * -cell_height

    grid_cells = [box(x, y - cell_height, x + cell_width, y) for y in y_coords for x in x_coords]

    gdf_grid = gpd.GeoDataFrame(
        {
            "total_population": total_agg.flatten(),
            "exposed_population": exposed_agg.flatten(),
            "avg_flood_depth": avg_flood_depth.flatten(),
            "pct_cells_flooded": (cells_flooded.flatten() / (factor**2)) * 100,
            "pct_cells_higher_1m": (pixels_high.flatten() / (factor**2)) * 100,
            "pct_cells_higher_1.5m": (pixels_higher.flatten() / (factor**2)) * 100
        },
        geometry=grid_cells,
        crs=crs
    )

    # --- Clip to region/background ---
    region_bg = gpd.overlay(region, background, how="intersection") if background is not None else region.copy()

    # --- Redistribute population proportionally to area inside region ---
    gdf_grid = gdf_grid.reset_index(names="cell_id")
    intersections = gpd.overlay(gdf_grid, region_bg, how="intersection")
    intersections["intersect_area"] = intersections.geometry.area
    area_sum = intersections.groupby("cell_id")["intersect_area"].transform("sum")
    intersections["norm_fraction"] = intersections["intersect_area"] / area_sum

    # Redistribute full population across pieces (sum per cell preserved)
    for col in ["total_population", "exposed_population"]:
        intersections[col] = intersections.groupby("cell_id")[col].transform("first") * intersections["norm_fraction"]

    # Aggregate pieces back to one polygon per cell
    gdf_grid_masked = intersections.dissolve(by="cell_id", aggfunc="sum")
    gdf_grid_masked["geometry"] = intersections.dissolve(by="cell_id").geometry

    # Relative exposure
    gdf_grid_masked["relative_population"] = (
        gdf_grid_masked["exposed_population"] / gdf_grid_masked["total_population"] * 100
    )

    print(f"Total pop (original): {np.nansum(total_pop_array):,.2f}")
    print(f"Total pop (aggregated): {gdf_grid_masked['total_population'].sum():,.2f}")
    print(f"Diff: {np.nansum(total_pop_array) - gdf_grid_masked['total_population'].sum():,.2f}")
    print(f"Diff %: {((np.nansum(total_pop_array) - gdf_grid_masked['total_population'].sum()) / np.nansum(total_pop_array)) * 100:,.2f}")

    return gdf_grid_masked

# Function to calculate exposed population based on flood depth threshold
def exposed_population_raster(pop, flood_depth, threshold=0.15):
    pop = pop[0] if pop.ndim == 3 else pop
    mask = flood_depth > threshold
    return pop * mask

# --- Function to save raster only if it does not yet exist ---
def save_raster(array, out_path, transform, crs):
    # """Save raster only if it does NOT yet exist."""
    # if os.path.exists(out_path):
    #     print(f"✔ File already exists, skipping: {out_path}")
    #     return
    
    print(f"▶ Writing raster: {out_path}")
    profile = {
        "driver": "GTiff",
        "dtype": rasterio.float32,
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "crs": crs,
        "transform": transform,
        "compress": "deflate"
    }
    
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)

# --- Function to apply standard map formatting to Cartopy axes ---
def setup_map_axes(axes, region_utm, background_utm, flood_extent, subplot_labels=None, titles=None,
                   show_left_labels_only=True, label_offset=(0, 1.02), axis_labelsize=9,
                   subplot_labelsize=10, title_fontsize=10):
    """
    Apply standard map formatting to one or more Cartopy axes:
    region boundary, background, extent, gridlines, labels.
    """
    axes_arr = np.atleast_1d(axes)
    ncols = axes_arr.shape[-1] if axes_arr.ndim >= 2 else axes_arr.size
    nrows = axes_arr.shape[0] if axes_arr.ndim >= 2 else 1
    axes_flat = axes_arr.ravel()
    for i, ax in enumerate(axes_flat):
        background_utm.plot(ax=ax, color="#E0E0E0", zorder=0)
        region_utm.boundary.plot(ax=ax, edgecolor="black", linewidth=0.3)
        ax.set_extent(flood_extent, crs=ccrs.UTM(36, southern_hemisphere=True))

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {"size": axis_labelsize}
        gl.ylabel_style = {"size": axis_labelsize}
        if show_left_labels_only and i % ncols != 0:
            gl.left_labels = False
        if i // ncols < nrows - 1:
            gl.bottom_labels = False
        

        if subplot_labels and i < len(subplot_labels):
            ax.text(
                label_offset[0],
                label_offset[1],
                subplot_labels[i],
                transform=ax.transAxes,
                fontsize=subplot_labelsize,
                fontweight="bold",      
                va="bottom",
                ha="left",
            )
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=title_fontsize)



#%%
pop_arrays = {}
# --- Reproject to flood grid and redistribute population rasters over land ---
for year, path, source in [
                           (1975, pop_raster_path_GHSL_100m_1975, "GHSL_100m"),
                           (2015, pop_raster_path_GHSL_100m_2015, "GHSL_100m"),
                           (2020, pop_raster_path_GHSL_100m_2020, "GHSL_100m"),
                           (2015, pop_path_GLOPOP_SG_1km, "GLOPOP"),
                        ]:
    pop_arrays[year, source] = reproject_and_redistribute_population_over_land(
        pop_path=path, land_gdf=background_utm, flood_crs=flood_grid_crs, flood_transform=flood_grid_transform,
        flood_shape=flood_grid_shape, region=region, year=year, source=source,
        out_raster_path=(join(BASE_DATA_PATH,f"preprocessed/population_{source}_{year}_region_regrid.tif"))) 

#%%
# Calculate exposed population for each dataset and store in results dictionary
sources = ["GHSL_100m", "GLOPOP"]
results = {s: {} for s in sources}
threshold = 0.15   # or 0.15

for (year, source) in pop_arrays.keys():
    pop = pop_arrays[year, source]
    results[source].setdefault(year, {})
    F_map = exposed_population_raster(pop, hmax_F, threshold)
    results[source][year]["F_raster_exposed"] = F_map
    results[source][year]["F_total_exposed"] = np.nansum(F_map)
    results[source][year]["pop_total"] = pop.sum()
    
rows = []

for source in results:
    for year in results[source]:
        rows.append({
            "source": source,
            "year": year,
            "total_population": results[source][year]["pop_total"],
            "F_exposed": results[source][year]["F_total_exposed"],
            })

df = pd.DataFrame(rows).sort_values(["year", "source"], inplace=True)
df


#%%
# Comparing GHSL 100m and GLOPOP data for 2015 and 2020
diff_exposed_2015 = (df['F_exposed'][df['source'] == 'GLOPOP'].values[0] / df['F_exposed'][(df['source'] == 'GHSL_100m') & (df['year'] == 2015)].values[0]) * 100
diff_exposed_2020 = (df['F_exposed'][df['source'] == 'GLOPOP'].values[0] / df['F_exposed'][(df['source'] == 'GHSL_100m') & (df['year'] == 2020)].values[0]) * 100

print(f"Difference in exposed population between GLOPOP and GHSL 100m for 2015: {diff_exposed_2015:.2f} %")
print(f"Difference in exposed population between GLOPOP 2015 and GHSL 100m for 2020: {diff_exposed_2020:.2f} %")


#%%
# ============================================================================================ #
# ======================= Compute total exposed population per district ====================== #
# ============================================================================================ #
# --- zonal statistics helper functions ---
# Function to calculate zonal sum of raster values within geometries
def zonal_sum(arr, geoms, transform):
    return [np.nansum(arr[features.geometry_mask([g], arr.shape, transform, invert=True)])
            for g in geoms]

# Function to calculate zonal sum for full districts
def zonal_sum_full(src, gdf):
        # --- 1. CHECK CRS ---
        if gdf.crs is None:
            raise ValueError("GeoDataFrame has no CRS defined.")

        if src.crs is None:
            raise ValueError("Raster has no CRS defined.")

        # --- 2. REPROJECT (IF NEEDED) ---
        if gdf.crs != src.crs:
            print(f"  ↪ Reprojecting {gdf.crs} → {src.crs}")
            gdf_native = gdf.to_crs(src.crs)
        else:
            gdf_native = gdf

        values = []
        for geom in gdf_native.geometry:
            out_img, _ = mask(src, [mapping(geom)], crop=True, nodata=src.nodata)
            arr = out_img[0].astype("float32")

            # handle nodata
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[arr <= 0] = np.nan

            values.append(np.nansum(arr))

        return values


# --- Inputs ---
pop_2020_path = pop_raster_path_GHSL_100m_2020
pop_2020_fine = pop_arrays[(2020, "GHSL_100m")]
exp_2020 = results["GHSL_100m"][2020]["F_raster_exposed"]

# --- 1) FLOOD GRID ZONAL STATS ---
adm3_full = (gpd.overlay(districts_adm3.to_crs(shapefile_sofala.crs), shapefile_sofala,
                                   how="intersection").to_crs(region.crs))

adm3_touch = adm3_full[adm3_full.intersects(region.union_all())]

# Remove districts outside study region
drop_districts = ["Muanza", "Gororngosa-Sede", "Galinha"]
adm3 = adm3_touch[~adm3_touch['NAME_3'].isin(drop_districts)]

# Calculate total population and exposed population per district using both the original raster and the regridded raster
adm3["pop_total"] = zonal_sum_full(rasterio.open(pop_GHSL_100m_2020), adm3)
adm3["pop_total_fine"] = zonal_sum(pop_2020_fine, adm3.geometry, flood_grid_transform)
adm3["pop_exposed"]    = zonal_sum(exp_2020, adm3.geometry, flood_grid_transform)

adm3["pop_total"] = adm3["pop_total"].apply(lambda x: f"{x:,.0f}")


#%%
# TABLE S3: Exposed population summary per district
df_district_summary = adm3[['NAME_3', 'pop_total', 'pop_total_fine', 'pop_exposed']].copy()
df_district_summary.columns = ['District', 'Total Pop', 'Total Pop within Region', 'Exposed Pop (Factual)']

# df_district_summary.to_csv("../results/tables_S02_S03.csv", index=False)
df_district_summary



#%%
gdf_pop_2020_exposed_F_coarse  = aggregate_pop(pop_arrays[(2020, "GHSL_100m")], hmax_F, 
                                               flood_grid_transform, flood_grid_crs, 
                                               region_utm, background_utm)
gdf_exposed_F = gdf_pop_2020_exposed_F_coarse.copy()
# --- Mask background ---
mask_poly = Polygon([(34.9, -20.3), (36, -20.3), (36, -19.9), (34.9, -19.9)])
bg_filtered = background.copy()
bg_filtered["geometry"] = bg_filtered.geometry.difference(mask_poly)
bg_filtered_utm = bg_filtered.to_crs(flood_grid_crs)

# Plot settings
norm_pop = PowerNorm(gamma=0.5, vmin=0, vmax=gdf_exposed_F["total_population"].max())
cmap_pop = mcolors.LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#FC6F37"])
norm_pop_exposed = PowerNorm(gamma=0.5, vmin=0, vmax=gdf_exposed_F["exposed_population"].max())
cmap_pop_exposed = mcolors.LinearSegmentedColormap.from_list(
"white_blue_purple", ["#ffffff", "#67CBE4", "#3B1F8C"])

#%% 
# FIGURE S1: Estimated total population and exposed population per district (coarse grid)
# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300,
                         subplot_kw={"projection": ccrs.UTM(36, southern_hemisphere=True)},
                         constrained_layout=True)

setup_map_axes(axes, region_utm, background_utm, flood_extent,
               subplot_labels=["(a)", "(b)"],
               titles=["Factual total population", "Factual exposed population"])

for ax in axes:
    background_utm.plot(ax=ax, color="#E0E0E0", zorder=0)
    bg_filtered_utm.boundary.plot(ax=ax, color="#B0B0B0", linewidth=0.5, zorder=1)
    adm3.boundary.plot(ax=ax, color="#4E4E4E", linewidth=1, zorder=10)

# --- Maps ---
gdf_exposed_F.loc[gdf_exposed_F["total_population"] == 0, "total_population"] = np.nan
gdf_exposed_F.plot(column="total_population", cmap=cmap_pop,
                   edgecolor="grey", norm=norm_pop,
                   linewidth=0.2, ax=axes[0], legend=False,
                   zorder=2, rasterized=True,
                   missing_kwds={"color": "none", "edgecolor": "none"})

gdf_exposed_F.loc[gdf_exposed_F["exposed_population"] == 0, "exposed_population"] = np.nan
gdf_exposed_F.plot(column="exposed_population", cmap=cmap_pop_exposed,
                   edgecolor="grey", norm=norm_pop_exposed,
                   linewidth=0.2, ax=axes[1], legend=False,
                   zorder=2, rasterized=True,
                   missing_kwds={"color": "none", "edgecolor": "none"})

# --- Offsets ---
offset_names = {
    "Nhamatanda": (15000, -30000),
    "Estaquinha": (15000, -10000),
    "Mafambisse": (3500, -4000),
    "Cidade Da Beira": (0, -1000),
    "Buzi": (20000, -3500),
    "Sofala": (5000, 10000),
}

offset_total = {
    "Nhamatanda": (18000, -26500),
    "Estaquinha": (20000, -5500),
    "Cidade Da Beira": (0, 2500),
    "Buzi": (20000, 0),
    "Sofala": (5000, 13000),
}

offset_exposed = {
    "Nhamatanda": (18000, -26500),
    "Estaquinha": (20000, -5500),
    "Cidade Da Beira": (0, 2500),
    "Buzi": (20000, 0),
    "Sofala": (5000, 13000),
}

# --- Labels ---
for _, row in adm3.iterrows():
    x0 = row.geometry.centroid.x
    y0 = row.geometry.centroid.y

    def fmt(v):
        if pd.isna(v):
            return ""
        v = int(v)
        if v < 100:
            v = int(round(v, -1))
        elif v < 1000:
            v = int(round(v, -2))
        else:
            v = int(round(v, -3)) 
        return f"{v:,}"

    # --- panel A: total ---
    dx, dy = offset_total.get(row["NAME_3"], (0, 0))
    axes[0].text(x0 + dx, y0 + dy,
                 fmt(row["pop_total_fine"]),
                 ha="center", va="center",
                 fontsize=7, color="black",
                 fontweight="bold", zorder=11)

    # --- panel B: exposed ---
    dx, dy = offset_exposed.get(row["NAME_3"], (0, 0))
    axes[1].text(x0 + dx, y0 + dy,
                 fmt(row["pop_exposed"]),
                 ha="center", va="center",
                 fontsize=7, color="black",
                 fontweight="bold", zorder=11)

    # --- name labels (FIXED) ---
    nx, ny = offset_names.get(row["NAME_3"], (0, -4000))

    for ax in axes:
        ax.text(x0 + nx, y0 + ny,
                row["NAME_3"],
                ha="center", va="center",
                fontsize=7, color="black",
                fontweight="bold", zorder=11)

# --- colorbars ---
sm = ScalarMappable(cmap=cmap_pop, norm=norm_pop)
sm._A = []
cbar0 = fig.colorbar(sm, ax=axes[0], shrink=0.6)
cbar0.set_label("Aggregated population (×10³ people)", fontsize=9)
cbar0.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))

sm = ScalarMappable(cmap=cmap_pop_exposed, norm=norm_pop_exposed)
sm._A = []
cbar1 = fig.colorbar(sm, ax=axes[1], shrink=0.6)
cbar1.set_label("Aggregated exposed population (×10³ people)", fontsize=9)
cbar1.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))

# fig.savefig("../figures/fS01.png", dpi=300, bbox_inches='tight')
# fig.savefig("../figures/fS01.pdf", dpi=300, bbox_inches='tight')



# %%
# ======================================================================================
# ============================= SOCIO-ECONOMIC VALIDATION ==============================
# ======================================================================================
# Comparing socio-economic characteristics per district
gdf_pop = gpd.GeoDataFrame(df_pop_charac, 
                           geometry=gpd.points_from_xy(df_pop_charac["x"], df_pop_charac["y"]),
                           crs="EPSG:4326")
adm1 = shapefile_sofala.to_crs(flood_grid_crs).copy()
adm3_glopop = adm3.to_crs(gdf_pop.crs)
adm1_glopop = adm1.to_crs(gdf_pop.crs)

gdf_pop_adm3 = gpd.sjoin(gdf_pop, adm3_glopop[["NAME_3", "geometry"]], how="left", predicate="within")
gdf_pop_adm1 = gpd.sjoin(gdf_pop, adm1_glopop[["NAME_1", "geometry"]], how="left", predicate="within")

gdf_pop_adm3_filt = gdf_pop_adm3[gdf_pop_adm3["NAME_3"].notna()]


# Compute rural and urban ratios
rural_ratio_sofala = (gdf_pop_adm1['RURAL'] == 1).sum() / ((gdf_pop_adm1['RURAL'] == 1) + (gdf_pop_adm1['RURAL'] == 0)).sum()
urban_ratio_sofala = (gdf_pop_adm1['RURAL'] == 0).sum() / ((gdf_pop_adm1['RURAL'] == 1) + (gdf_pop_adm1['RURAL'] == 0)).sum()

rural_ratio_studyarea = (gdf_pop_adm3_filt['RURAL'] == 1).sum() / ((gdf_pop_adm3_filt['RURAL'] == 1) + (gdf_pop_adm3_filt['RURAL'] == 0)).sum()
urban_ratio_studyarea = (gdf_pop_adm3_filt['RURAL'] == 0).sum() / ((gdf_pop_adm3_filt['RURAL'] == 1) + (gdf_pop_adm3_filt['RURAL'] == 0)).sum()

print(f"Rural ratio in Sofala province: {rural_ratio_sofala:.2%}")
print(f"Urban ratio in Sofala province: {urban_ratio_sofala:.2%}")
print(f"Rural ratio in study area: {rural_ratio_studyarea:.2%}")
print(f"Urban ratio in study area: {urban_ratio_studyarea:.2%}")


#%%
# FOR TABLE S4: Socio-economic characteristics of the population per settlement type
df = gdf_pop_adm1.copy()

# Define all groupings 
GROUPS = {
    "WEALTH": {
        "Low (1–2)": [1, 2],
        "Middle (3)": [3],
        "High (4–5)": [4, 5],
    },
    "EDUC": {
        "Low (<primary)": [1, 2],
        "Secondary (3)": [3],
        "High (4–5)": [4, 5],
    },
    "HHSIZE_CAT": {
        "Single (1)": "==1",
        "Multiple (>1)": ">1",
    },
    "GENDER": {
        "Male": 1,
        "Female": 0,
    },
    "AGE": {
        "Young/Old": [1, 2, 8],
        "Middle": [3, 4, 5, 6, 7],
    }}

def apply_mask(df, col, rule):
    if isinstance(rule, list):
        return df[col].isin(rule)
    elif isinstance(rule, str):
        return eval(f"(df[col] {rule})")
    else:
        return df[col] == rule



# Compute summary table
rows = []

for var, groups in GROUPS.items():
    for label, rule in groups.items():
        mask = apply_mask(df, var, rule)
        total = mask.sum()

        rural_total = (df["RURAL"] == 1).sum()
        urban_total = (df["RURAL"] == 0).sum()

        rural_ratio = ((mask & (df["RURAL"] == 1)).sum() / rural_total) if rural_total else np.nan
        urban_ratio = ((mask & (df["RURAL"] == 0)).sum() / urban_total) if urban_total else np.nan

        rows.append({
            "variable": var,
            "category": label,
            "count": total,
            "rural_ratio": rural_ratio,
            "urban_ratio": urban_ratio})

summary = pd.DataFrame(rows)
summary["rural_ratio"] = summary["rural_ratio"].map(lambda x: f"{x:.2%}")
summary["urban_ratio"] = summary["urban_ratio"].map(lambda x: f"{x:.2%}")

print(summary)


#%%
# For comparison with Census 2017 data - calculate population growth rate from 2015 to 2017 for Mozambique
# from https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fpopulation.un.org%2Fwpp%2Fassets%2FExcel%2520Files%2F1_Indicator%2520(Standard)%2FEXCEL_FILES%2F1_General%2FWPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx&wdOrigin=BROWSELINK
# or: https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Most%20used
total_pop_MZ_2015 = 26162000  # from World Population Proespects 
total_pop_MZ_2017 = 27741000  # from World Population Proespects

growth_rate_2015_to_2017 = ((total_pop_MZ_2017 - total_pop_MZ_2015) / total_pop_MZ_2015) * 100

print(f"Population growth rate in Mozambique from 2015 to 2017: {growth_rate_2015_to_2017:.2f} %")



