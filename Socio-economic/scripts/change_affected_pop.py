#%% run this script using the pixi environmental compass-socio
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join
import rasterio
from rasterio import features
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from rasterio.features import rasterize
import cartopy.crs as ccrs
from pyproj import Transformer
from rasterio.mask import mask
from shapely.geometry import box, Polygon
import rioxarray as rxr 
from tqdm import tqdm
from scipy.signal import find_peaks
from affine import Affine


# ====== HELPER FUNCTIONS ======
# get extent from raster transform
def get_extent(transform, width, height):
    left = transform[2]
    right = left + width * transform[0]
    top = transform[5]
    bottom = top + height * transform[4]
    return [left, right, top, bottom]

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

# --- Function to setup map axes with consistent formatting ---
def setup_map_axes(
    axes,
    region_utm,
    background_utm,
    flood_extent,
    subplot_labels=None,
    titles=None,

    # layout controls
    show_left_labels_only=True,
    show_gridlabels=True,

    # styling
    axis_labelsize=9,
    subplot_labelsize=10,
    title_fontsize=10,

    # optional extra layers
    background_outside_box=None,

    # annotation control
    add_outside_background=True,
    add_grid=True,
    add_subplot_labels=True,
    add_titles=True,
):
    """
    Standardized Cartopy map formatting for multi-panel figures.
    """

    axes_arr = np.atleast_1d(axes)
    ncols = axes_arr.shape[-1] if axes_arr.ndim >= 2 else axes_arr.size
    nrows = axes_arr.shape[0] if axes_arr.ndim >= 2 else 1
    axes_flat = axes_arr.ravel()

    for i, ax in enumerate(axes_flat):

        # -----------------------
        # Base map layers
        # -----------------------
        background_utm.plot(ax=ax, color="#E0E0E0", zorder=0)
        region_utm.boundary.plot(ax=ax, edgecolor="black", linewidth=0.3)

        ax.set_extent(
            flood_extent,
            crs=ccrs.UTM(36, southern_hemisphere=True),
        )

        # -----------------------
        # Outside background (optional)
        # -----------------------
        if add_outside_background and background_outside_box is not None:
            background_outside_box.plot(
                ax=ax,
                color="#E0E0E0",
                transform=ccrs.PlateCarree(),
                zorder=0,
            )
            background_outside_box.boundary.plot(
                ax=ax,
                color="#818181",
                linewidth=0.2,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # -----------------------
        # Gridlines
        # -----------------------
        if add_grid and show_gridlabels:
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.5,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )

            gl.right_labels = False
            gl.top_labels = False
            gl.xlabel_style = {"size": axis_labelsize}
            gl.ylabel_style = {"size": axis_labelsize}

            if show_left_labels_only and i % ncols != 0:
                gl.left_labels = False

            if i // ncols < nrows - 1:
                gl.bottom_labels = False

        # -----------------------
        # Subplot labels
        # -----------------------
        if add_subplot_labels and subplot_labels and i < len(subplot_labels):
            ax.text(
                0,
                1.02,
                subplot_labels[i],
                transform=ax.transAxes,
                fontsize=subplot_labelsize,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        # -----------------------
        # Titles
        # -----------------------
        if add_titles and titles and i < len(titles):
            ax.set_title(titles[i], fontsize=title_fontsize)

# --- Function to convert extent from lat/lon to UTM (for consistent plotting and box addition) ---
def extent_to_utm(extent):
    xmin, xmax, ymin, ymax = extent
    x1, y1 = transformer.transform(xmin, ymin)  # bottom-left
    x2, y2 = transformer.transform(xmax, ymax)  # top-right
    return [x1, x2, y1, y2]

# --- Function to add a box to an axis given an extent (in UTM) ---
def add_box(ax, extent):
    xmin, xmax, ymin, ymax = extent
    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=1,
        edgecolor="black",
        facecolor="none",
        # transform=ccrs.PlateCarree(),
        zorder=10
    )
    ax.add_patch(rect)

# --- Function to add reference locations to the map ---
def add_reference_locations(ax):

    locations = [
        (34.862, -19.833, "Beira", (34.852, -19.89)),
        (34.43,  -19.89,  "Buzi River", (34.44, -19.87)),
        (34.543, -19.545, "Pungwe River", (34.554, -19.52)),
    ]

    for lon, lat, label, (tx, ty) in locations:

        ax.plot(
            lon,
            lat,
            marker="o",
            color="black",
            markersize=3,
            markeredgecolor="white",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        txt = ax.text(
            tx,
            ty,
            label,
            fontsize=8,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        txt.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ])

    ax.text(
        34.985,
        -19.66,
        "Beira municipality",
        fontsize=7.2,
        ha="center",
        va="center",
        style="italic",
        color="#5C5C5C",
        transform=ccrs.PlateCarree(),
        zorder=4,
    )


### functions for population analysis ###
# --- Function to redistribute population over land pixels on flood grid ---
def reproject_and_redistribute_population_over_land(pop_path, land_gdf, flood_crs, flood_transform, flood_shape, province_geom=None, region=None, districts_adm3=None, districts_adm2=None, year=None, out_raster_path=None, source=None):    
    print(f"▶ Loading {year} population raster...")
    with rasterio.open(pop_path) as src:
        pop = src.read(1, masked=True)
        pop_affine = src.transform
        pop_crs = src.crs

        # Clip to province
        province_geom = province_geom.to_crs(src.crs)
        province_geom = [json.loads(province_geom.to_json())["features"][0]["geometry"]]

        pop_sofala, transform_sofala = mask(src, province_geom, crop=True, nodata=src.nodata)

        # Dissolve districts into one geometry
        districts_adm3_single = districts_adm3.dissolve().reset_index(drop=True)
        districts_adm3_single = districts_adm3_single.to_crs(src.crs)
        districts_adm3_geom   = [districts_adm3_single.geometry.iloc[0].__geo_interface__]

        districts_adm2_single = districts_adm2.dissolve().reset_index(drop=True)
        districts_adm2_single = districts_adm2_single.to_crs(src.crs)
        districts_adm2_geom   = [districts_adm2_single.geometry.iloc[0].__geo_interface__]

        pop_districts_adm3, pop_affine_districts_adm3 = mask(src, districts_adm3_geom, crop=True, nodata=src.nodata)
        pop_districts_adm2, pop_affine_districts_adm2 = mask(src, districts_adm2_geom, crop=True, nodata=src.nodata)

        if out_raster_path is not None and os.path.exists(out_raster_path):
            print(f"▶ Loading existing raster from {out_raster_path}")
            with rasterio.open(out_raster_path) as src:
                pop_fine = src.read(1)
                return pop_fine, pop_sofala, transform_sofala, pop_districts_adm3, pop_affine_districts_adm3, pop_districts_adm2, pop_affine_districts_adm2

        # Clip to region if provided
        if region is not None:
            region_wsg = region.to_crs(src.crs)
            region_geom = [json.loads(region_wsg.to_json())["features"][0]["geometry"]] 
            pop, pop_affine = rasterio.mask.mask(src, region_geom, crop=True, nodata=src.nodata)

        # --- clean raster properly (IMPORTANT) ---
        pop = pop.filled(np.nan) if np.ma.isMaskedArray(pop) else pop.squeeze()

        # GHSL nodata handling (covers -200, -9999, etc.)
        if src.nodata is not None:
            pop[pop == src.nodata] = np.nan

        # remove all invalid population values
        pop[pop <= 0] = np.nan

        pop = pop.squeeze()

    # Prepare empty high-resolution array
    pop_fine = np.zeros(flood_shape, dtype=np.float32)

    # Rasterize land mask to flood grid (True for land)
    land_mask = features.rasterize(
        [(geom, 1) for geom in land_gdf.geometry],
        out_shape=flood_shape,
        transform=flood_transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    print("  ✔ Land mask created on flood grid.")

    # Loop through each coarse pixel
    print("▶ Redistributing population to fine grid...")
    for row in tqdm(range(pop.shape[0]), desc="  Processing coarse cells"):
        for col in range(pop.shape[1]):
            pop_value = pop[row, col]
            if np.isnan(pop_value) or pop_value <= 0:
                continue

            # Get coarse pixel bounds (in coarse CRS)
            x_min, y_max = pop_affine * (col, row)
            x_max, y_min = pop_affine * (col + 1, row + 1)
            coarse_bounds = box(x_min, y_min, x_max, y_max)

            # Transform to flood CRS
            coarse_bounds_flood = gpd.GeoSeries([coarse_bounds], crs=pop_crs).to_crs(flood_crs).iloc[0]

            # Rasterize this coarse cell footprint to flood grid
            coarse_mask = features.rasterize(
                [(coarse_bounds_flood, 1)],
                out_shape=flood_shape,
                transform=flood_transform,
                fill=0,
                # all_touched=True,
                dtype=np.uint8
            ).astype(bool)

            # Identify land pixels within this coarse cell
            valid_mask = coarse_mask & land_mask
            n_valid = valid_mask.sum()

            # Distribute population evenly over valid pixels
            if n_valid > 0:
                # pop_fine[valid_mask] += pop_value / n_valid # old resulting in "people behind the comma"

                # --- Integer redistribution (whole people only) ---
                P = int(round(pop_value))
                N = n_valid

                base = P // N
                remainder = P % N

                # convert mask indices to linear index list
                valid_indices = np.where(valid_mask)

                # assign the base number to all pixels
                pop_fine[valid_indices] += base

                # assign remainder one-by-one to the first 'remainder' pixels
                # (or shuffle if you prefer randomness)
                if remainder > 0:
                    # deterministic: first R indices
                    # if random desired: uncomment the shuffle line below
                    perm = np.random.permutation(len(valid_indices[0]))
                    chosen = perm[:remainder]
                    # chosen = np.arange(remainder)
                    pop_fine[valid_indices[0][chosen], valid_indices[1][chosen]] += 1


    # else: all water → skip or optionally add to nearest land (not done here)
    total_input_pop = float(np.nansum(pop))
    total_output_pop = float(pop_fine.sum())
    diff = abs(total_output_pop - total_input_pop)
    rel_diff = diff / total_input_pop * 100

    # --- Validation printout ---
    print("  ✔ Redistribution done.")
    print(f"  🔹 Input population:  {total_input_pop:,.0f}")
    print(f"  🔹 Output population: {total_output_pop:,.0f}")
    print(f"  🔹 Difference:        {diff:,.2f} ({rel_diff:.4f} %)")

    if rel_diff > 0.01:
        print("  ⚠ WARNING: Population not perfectly preserved — check CRS or mask alignment!")

    print(f"  ✔ Redistribution done. Total population preserved: {pop_fine.sum():,.0f}")
    
    # Optional: save the result as a GeoTIFF
    if out_raster_path is not None:
        print(f"▶ Saving redistributed population raster to {out_raster_path}")
        H, W = pop_fine.shape
        a, b, c, d, e, f, *_ = flood_transform  # unpack affine

        if e > 0:
            print("  ⚠ Detected positive y-resolution in transform → fixing for QGIS")

            # 1) flip array vertically
            pop_fine_to_write = np.flipud(pop_fine)

            # 2) fix y-scale sign and y-origin
            new_e = -abs(e)
            new_f = f + e * (H - 1)

            fixed_transform = Affine(a, b, c, d, new_e, new_f)

        else:
            pop_fine_to_write = pop_fine
            fixed_transform = flood_transform

        # ------------------------------------------------------------------

        new_profile = {
            "driver": "GTiff",
            "dtype": rasterio.float32,
            "count": 1,
            "height": H,
            "width": W,
            "crs": flood_crs,
            "transform": fixed_transform,
            "compress": "deflate"
        }

        with rasterio.open(out_raster_path, "w", **new_profile) as dst:
            dst.write(pop_fine_to_write, 1)

    return pop_fine, pop_sofala, transform_sofala, pop_districts_adm3, pop_affine_districts_adm3, pop_districts_adm2, pop_affine_districts_adm2

# --- Function to link population raster to flood depth as DataFrame ---
def pop_raster_to_gdf(pop_array, flood_array, transform, year, climate, export_df=True, export_path=None):
    print("Linking population raster to flood depth as DataFrame...")

    # Flatten arrays
    pop_flat = pop_array.ravel()
    flood_flat = flood_array.ravel()

    # Mask zero-pop cells
    mask = (pop_flat > 0) & (flood_flat > 0)
    pop_vals = pop_flat[mask]
    flood_vals = flood_flat[mask]

    # Pixel coordinates (centers)
    rows, cols = np.indices(pop_array.shape)
    xs, ys = transform * (cols.ravel()[mask] + 0.5, rows.ravel()[mask] + 0.5)

    df = pd.DataFrame({
        "population": pop_vals,
        "flood_depth": flood_vals,
        "x": xs,
        "y": ys
    })

    if export_df:
        file_name = f"df_pop_{year}_{climate}.csv"
        df.to_csv(join(export_path, file_name), index=False)
        print(f"▶ Exported DataFrame to {join(export_path, file_name)}")

    return df

# --- Function to aggregate population and flood depth to coarser grid ---
# Problem is that population is either lost when area weighted averaging or overcounted when summing and redistributing.
def aggregate_pop(total_pop_array, flood_raster, transform, crs, region=None, background=None, factor=100):
    print("Aggregating population raster to coarser grid polygons...")

    # Pixel size from transform
    pixel_width = transform.a
    pixel_height = -transform.e
    pixel_area = pixel_width * -pixel_height  

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
    flooded_area = cells_flooded * pixel_area
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
            "nr_cells_flooded": cells_flooded.flatten(),
            "area_flooded": flooded_area.flatten(),
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

# --- Function to print exposure statistics ---
def print_exposure_stats(label, pop, exposed):
    print(f"\n{label}")
    print(f"Total population in region: {np.nansum(pop):,.0f}")
    print(f"Total exposed people: {np.nansum(exposed):,.0f}")
    print(
        f"Exposed people percentage of total population: "
        f"{100 * np.nansum(exposed) / np.nansum(pop):.2f}%"
    )


# =======================================================================================================================
# =================================================== START OF CODE =====================================================
# =======================================================================================================================
# ===== CONFIGURATION =====
EVENT_NAME = "Idai"
BASE_DATA_PATH = Path("p:/11210471-001-compass/01_Data")
BASE_RUN_PATH = Path("p:/11210471-001-compass/03_Runs/sofala/Idai")
SCENARIO_PATH_F = "event_tp_era5_hourly_zarr_CF0_GTSMv41_CF0_era5_hourly_spw_IBTrACS_CF0" # factual
SCENARIO_PATH_CF = "event_tp_era5_hourly_zarr_CF-6_GTSMv41_CF-0.07_era5_hourly_spw_IBTrACS_CF-4" # counterfactual

# =======================================================================================================================
# ===== FLOOD MODEL FILES ==== 
# Base directory for the specific event and scenario
sfincs_dir_F  = BASE_RUN_PATH / "sfincs" / SCENARIO_PATH_F
sfincs_dir_CF = BASE_RUN_PATH / "sfincs" / SCENARIO_PATH_CF

# Read flood rasters
F_flooding = sfincs_dir_F / "plot_output" / "floodmap_15cm.tif"
CF_flooding = sfincs_dir_CF / "plot_output" / "floodmap_15cm.tif"
hmax_F_da = rxr.open_rasterio(F_flooding).squeeze("band", drop=True)  # if single-band
hmax_CF_da = rxr.open_rasterio(CF_flooding).squeeze("band", drop=True)  # if single-band
hmax_F = hmax_F_da.values
hmax_CF = hmax_CF_da.values
hmax_diff = hmax_F - hmax_CF

# Flood model subgrid and region geometry
sfincs_subgrid = join(sfincs_dir_F, "subgrid", "dep_subgrid.tif")

# --- Flood grid properties ---
with rasterio.open(sfincs_subgrid) as src:
    flood_grid_crs, flood_grid_transform, flood_grid_shape = src.crs, src.transform, (src.height, src.width)
flood_extent = get_extent(flood_grid_transform, flood_grid_shape[1], flood_grid_shape[0])

# --- Setup region ---
region_wsg84 = gpd.read_file(join(sfincs_dir_F, "gis/region.geojson")).to_crs("EPSG:4326")
region_geom_poly = region_wsg84.geometry.iloc[0]
region = region_wsg84.to_crs(flood_grid_crs)
region_geom = [json.loads(region.to_json())["features"][0]["geometry"]]

# =======================================================================================================================
# ===== GEOSPATIAL DATA =====
# background = gpd.read_file(join(BASE_DATA_PATH, "gis/case_study_region_background.geojson"), driver="GeoJSON")
# shapefile_sofala = gpd.read_file(join(BASE_DATA_PATH, "gis/sofala_province.shp"))  # from https://gadm.org/ and processed
# beira_district = gpd.read_file(join(BASE_DATA_PATH, "gis/Beira_region.shp")) # from https://gadm.org/ and processed
# districts_adm3 = gpd.read_file(join(BASE_DATA_PATH, "gis/sofala_districts_study_region.shp")) # from https://gadm.org/ and processed
# gdf = gpd.read_file(join(BASE_DATA_PATH, "gis/gadm41_MOZ_2.shp")).to_crs(region_wsg84.crs) # from https://gadm.org/ and processed
background = gpd.read_file(join(BASE_DATA_PATH, "sofala_geoms/sofala_region_background.geojson"), driver="GeoJSON")
shapefile_sofala = gpd.read_file(join(BASE_DATA_PATH, "sofala_geoms/sofala_province.shp"))  # from https://gadm.org/ and processed
beira_district = gpd.read_file(join(BASE_DATA_PATH, "sofala_geoms/Beira_region.shp")) # from https://gadm.org/ and processed
districts_adm3 = gpd.read_file(join(BASE_DATA_PATH, "sofala_geoms/sofala_districts_study_region.shp")) # from https://gadm.org/ and processed
gdf = gpd.read_file(join(BASE_DATA_PATH, "sofala_geoms/gadm41_MOZ_shp/gadm41_MOZ_2.shp")).to_crs(region_wsg84.crs) # from https://gadm.org/ and processed

districts_adm2 = gdf[gdf.intersects(region_geom_poly)].copy()

# Load urban polygons, reproject to match flood raster CRS and rasterize
gdf_urban = gpd.read_file("../data/GLOPOP-SG/urban_area.geojson")
gdf_urban = gdf_urban.to_crs(flood_grid_crs)
urban_mask = rasterize([(geom, 1) for geom in gdf_urban.geometry],
                       out_shape=hmax_F.shape,
                       transform=flood_grid_transform,
                       fill=0, dtype='uint8')

# --- Background layers for plotting ---
# Define a polygon to remove/mask out land layer (incorrect boundary over the ocean)
mask_poly = Polygon([(34.9,-20.3), (36,-20.3), (36,-19.9), (34.9,-19.9)])
bg_filtered = background.copy()
bg_filtered['geometry'] = bg_filtered.geometry.apply(lambda g: g.difference(mask_poly))

# Reproject background and region to flood grid CRS for consistent plotting
background_utm = background.to_crs(flood_grid_crs)
bg_filtered_utm = bg_filtered.to_crs(flood_grid_crs)
region_utm = region.to_crs(flood_grid_crs)
districts_adm3_utm = districts_adm3.to_crs(flood_grid_crs)
districts_adm2_utm = districts_adm2.to_crs(flood_grid_crs)
beira_utm = beira_district.to_crs(flood_grid_crs)

# Land boundary based of background
mask_box = box(34.8, -20.3, 35.3, -19.9)  # minx, miny, maxx, maxy
background_outside_box = background[~background.intersects(mask_box)]

# Remove districts that are not connecting to the region
drop_districts = ["Muanza", "Gororngosa-Sede", "Galinha"]
districts_adm3_filtered = districts_adm3_utm[~districts_adm3_utm['NAME_3'].isin(drop_districts)]

# =======================================================================================================================
# ===== POPULATION DATA =====
# population raster from the Global Human Settlement Layer (GHSL) database (original 100 m resolution)
# You can download the GHSL population rasters for 2020 and 1975 here: https://human-settlement.emergency.copernicus.eu/download.php?ds=pop
population_raster_path_2020 = Path(join(BASE_DATA_PATH, "population_data/GHSL_POP/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R12_C22.tif"))
population_raster_path_1975 = Path(join(BASE_DATA_PATH, "population_data/GHSL_POP/GHS_POP_E1975_GLOBE_R2023A_54009_100_V1_0_R12_C22.tif"))


#%%
# ============================================================================================ #
# ====================== Process population directly into flood grid ========================= #
# ============================================================================================ #
export_path = "../data/preprocessed/population/"
population_configs = [(1975, population_raster_path_1975, "GHSL_100m"), (2020, population_raster_path_2020, "GHSL_100m")]
flood_maps = {"F": hmax_F, "CF": hmax_CF}

# -------------------------------------------------------------------------------------------- #
# Reproject population to flood grid
# -------------------------------------------------------------------------------------------- #
pop_data = {}
for year, path, source in population_configs:
    (pop, pop_sofala, transform_sofala_land, adm3, adm3_affine, adm2, adm2_affine
     ) = reproject_and_redistribute_population_over_land(
        pop_path=path, land_gdf=background_utm, flood_crs=flood_grid_crs, flood_transform=flood_grid_transform,
        flood_shape=flood_grid_shape, province_geom=shapefile_sofala, region=region, districts_adm3=districts_adm3_filtered,
        districts_adm2=districts_adm2, year=year, source=source,
        out_raster_path=(join(BASE_DATA_PATH, f"population_data/downscaled/population_{source}_{year}_region_regrid.tif")))

    pop_data[year] = {
        "population": pop,
        "pop_sofala": pop_sofala,
        "transform_sofala_land": transform_sofala_land,
        "adm3": adm3,
        "adm3_affine": adm3_affine,
        "adm2": adm2,
        "adm2_affine": adm2_affine
    }

# -------------------------------------------------------------------------------------------- #
# Flood exposure calculations
# -------------------------------------------------------------------------------------------- #
flood_depth_gdfs = {}
exposed_rasters = {}
exposed_gdfs = {}
coarse_gdfs = {}

for year in pop_data:
    flood_depth_gdfs[year] = {}
    exposed_rasters[year] = {}
    exposed_gdfs[year] = {}
    coarse_gdfs[year] = {}

    pop = pop_data[year]["population"]

    for climate, hmax in flood_maps.items():
        gdf = pop_raster_to_gdf(pop, hmax, flood_grid_transform, year=year, climate=climate, export_df=True, export_path=export_path)
        
        flood_depth_gdfs[year][climate] = gdf
        exposed_rasters[year][climate] = np.where(hmax > 0, pop, 0)
        exposed_gdfs[year][climate] = gdf[gdf["flood_depth"] > 0]
        coarse_gdfs[year][climate] = aggregate_pop(pop, hmax, flood_grid_transform, flood_grid_crs, region_utm, background_utm)

# ============================================================================================ #
# Uniform population growth scenario
# ============================================================================================ #
pop_growth = (np.nansum(pop_data[2020]["population"]) - np.nansum(pop_data[1975]["population"])
              ) / np.nansum(pop_data[1975]["population"])
pop_uniform_2020 = pop_data[1975]["population"] * (1 + pop_growth)

print(f"Uniform population growth from 1975 to 2020 in case study region: "f"{100 * pop_growth:.2f}%")
print(f"{np.nansum(pop_uniform_2020):,.0f} people in 2020 with uniform growth")
print(f"{np.nansum(pop_data[2020]['population']):,.0f} people in 2020 actual")

uniform_exposed_rasters = {}
uniform_flood_depth_gdfs = {}
uniform_exposed_gdfs = {}

for climate, hmax in flood_maps.items():
    uniform_exposed_rasters[climate] = np.where(hmax > 0, pop_uniform_2020, 0)

    gdf = pop_raster_to_gdf(pop_uniform_2020, hmax, flood_grid_transform, year="2020_uniform", 
                            climate=climate, export_df=True, export_path=export_path)
    uniform_flood_depth_gdfs[climate] = gdf
    uniform_exposed_gdfs[climate] = gdf[gdf["flood_depth"] > 0]

# ============================================================================================ #
# Attribution statistics
# ============================================================================================ #
scenario_exposure = {
    "2020_F": np.nansum(exposed_rasters[2020]["F"]),
    "2020_CF": np.nansum(exposed_rasters[2020]["CF"]),
    "1975_F": np.nansum(exposed_rasters[1975]["F"]),
    "1975_CF": np.nansum(exposed_rasters[1975]["CF"]),
    "2020_uniform_F": np.nansum(uniform_exposed_rasters["F"]),
}

perct_attr_clim     = (scenario_exposure["2020_F"] - scenario_exposure["2020_CF"]) / scenario_exposure["2020_F"] * 100
perct_attr_pop      = ( scenario_exposure["2020_F"] - scenario_exposure["1975_F"]) / scenario_exposure["2020_F"] * 100
perct_attr_clim_pop = (scenario_exposure["2020_F"] - scenario_exposure["1975_CF"]) / scenario_exposure["2020_F"] * 100

# printing stats
print_exposure_stats("2020 Factual exposed population stats:", pop_data[2020]["population"], 
                     exposed_rasters[2020]["F"])
print_exposure_stats("2020 Counterfactual exposed population stats:", pop_data[2020]["population"], 
                     exposed_rasters[2020]["CF"])
print_exposure_stats("1975 Factual exposed population stats:", pop_data[1975]["population"], 
                     exposed_rasters[1975]["F"])
print_exposure_stats("1975 Counterfactual exposed population stats:", pop_data[1975]["population"], 
                     exposed_rasters[1975]["CF"])
print_exposure_stats("2020 UNIFORM exposed population stats:", pop_uniform_2020, uniform_exposed_rasters["F"])

print("\nOne-line attribution numbers:")
print(f"Exposed population in 2020 Factual: " f"{scenario_exposure['2020_F']:,.0f}")
print(f"Exposed population attributable to climate change: " f"{scenario_exposure['2020_F'] - scenario_exposure['2020_CF']:,.0f} "
      f"({perct_attr_clim:.2f}%)")
print(f"Exposed population attributable to population change (2020-1975): " f"{scenario_exposure['2020_F'] - scenario_exposure['1975_F']:,.0f} "
      f"({perct_attr_pop:.2f}%)")
print(f"Exposed population attributable to population change and climate change: " f"{scenario_exposure['2020_F'] - scenario_exposure['1975_CF']:,.0f} "
      f"({perct_attr_clim_pop:.2f}%)")
print(f"Exposed population attributable to population change "
      f"(uniform growth): " f"{scenario_exposure['2020_F'] - scenario_exposure['2020_uniform_F']:,.0f} "
      f"({100 * (scenario_exposure['2020_F'] - scenario_exposure['2020_uniform_F']) / scenario_exposure['2020_F']:.2f}%)")
print(f"Population growth from 1975 to 2020 in the region: " f"{np.nansum(pop_data[2020]['population']) - np.nansum(pop_data[1975]['population']):,.0f} "
     f"({100 * (np.nansum(pop_data[2020]['population']) - np.nansum(pop_data[1975]['population'])) / np.nansum(pop_data[1975]['population']):.2f}%)")


#%% ============================================================================================ # 
# ================== Plot distribution of flood depth per exposed population =================== #
# ============================================================================================== #
def compute_cdf_and_bins(gdf, bins, depth_col="flood_depth", pop_col="population"):
    """Return population counts per flood depth bin."""
    flood = gdf[depth_col].values
    pop   = gdf[pop_col].values

    # Mask invalid
    mask = ~np.isnan(flood) & (pop > 0) # select only cells with flooding AND population
    flood, pop = flood[mask], pop[mask]

    # Binned population
    pop_by_depth = pd.Series(pop).groupby(pd.cut(flood, bins)).sum()

    return pop_by_depth

# Flood depth bins    
bins_fine = np.arange(0, 3.5 + 0.02, 0.01)
bins_coarse = np.arange(0, 3.5 + 0.2, 0.1)

bin_centers = bins_fine[:-1] + np.diff(bins_fine) / 2
bin_centers_coarse = bins_coarse[:-1] + np.diff(bins_coarse) / 2

# Flood depth categories
depth_masks = {"Low": (bins_fine[:-1] >= 0.15) & (bins_fine[:-1] < 0.5),
               "Medium": (bins_fine[:-1] >= 0.5) & (bins_fine[:-1] < 1.5),
               "High": bins_fine[:-1] >= 1.5}

# Scenarios
scenario_gdfs = {"2020_F": exposed_gdfs[2020]["F"],
                 "2020_CF": exposed_gdfs[2020]["CF"],
                 "1975_F": exposed_gdfs[1975]["F"],
                 "1975_CF": exposed_gdfs[1975]["CF"],
                 "2020_uniform_F": uniform_exposed_gdfs["F"]}

# Population distributions per flood depth
depth_dist_fine = {scenario: compute_cdf_and_bins(gdf, bins_fine)
                    for scenario, gdf in scenario_gdfs.items()}

depth_dist_coarse = {scenario: compute_cdf_and_bins(gdf, bins_coarse)
                     for scenario, gdf in scenario_gdfs.items() if scenario != "2020_uniform_F"}

# Attribution differences
diffs = {"Climate": depth_dist_fine["2020_F"].values - depth_dist_fine["2020_CF"].values,
         "Population": depth_dist_fine["2020_F"].values - depth_dist_fine["1975_F"].values,
         "Climate + Population": depth_dist_fine["2020_F"].values - depth_dist_fine["1975_CF"].values}

# Absolute change per flood depth category
data_abs_diff = np.array([[diff[mask].sum() for diff in diffs.values()]
                          for mask in depth_masks.values()])

# Relative change per flood depth bin
change_per_depth = pd.DataFrame({
    "Factual": depth_dist_coarse["2020_F"],
    "CF_climate": depth_dist_coarse["2020_CF"],
    "CF_population": depth_dist_coarse["1975_F"],
    "CF_climate_population": depth_dist_coarse["1975_CF"]})

for scenario in ["CF_climate", "CF_population", "CF_climate_population"]:
    change_per_depth[f"Rel_change_{scenario}"] = (
        (change_per_depth["Factual"] - change_per_depth[scenario])
        / change_per_depth["Factual"] * 100)

# Relative attributable exposed population by flood depth category
baseline = depth_dist_fine["2020_F"].values

data_attr = np.array([[np.nansum(diff[mask]) / np.nansum(baseline[mask]) * 100
                       for diff in diffs.values()] for mask in depth_masks.values()])    

# Plotting settings
colours = ["#00B050", "#1E2E57", "#28C2E9", "#9B59B6"]

x_bg = np.linspace(0, 3.5, 500)

low_mask_bg = (x_bg >= 0.15) & (x_bg < 0.5)
mid_mask_bg = (x_bg >= 0.5) & (x_bg < 1.5)
high_mask_bg = x_bg >= 1.5


#%%
# ==================================================================================================== #
# =========================== Plotting exposed population per flood depth ============================ #
# ==================================================================================================== #
# FIGURE 2: Maps of factual exposed population and changes in flood impact drivers
def plot_factual_and_driver_changes_extent(gdf_pop_exposed_F_coarse, gdf_pop_exposed_CF_clim_coarse, gdf_pop_exposed_CF_pop_coarse, 
                                             region_utm, background_utm, flood_extent):
    # Data preparation for plotting
    gdf_F = gdf_pop_exposed_F_coarse.copy()
    gdf_F.loc[gdf_F["total_population"] == 0] = np.nan
    gdf_CF_pop = gdf_pop_exposed_CF_pop_coarse.copy()
    gdf_CF_pop.loc[gdf_CF_pop["total_population"] == 0] = np.nan
    gdf_CF_clim = gdf_pop_exposed_CF_clim_coarse.copy()
    gdf_F["change_area_flooded"] = gdf_F["area_flooded"] - gdf_CF_clim["area_flooded"]
    gdf_F["attr_flood_extent"] = gdf_F["change_area_flooded"].sum() / gdf_F["area_flooded"].sum() * 100
    gdf_F["change_in_population"] = gdf_F["total_population"] - gdf_CF_pop["total_population"]
    
    # colour maps and norms
    norm_pop_exposed = PowerNorm(gamma=0.5, vmin=0, vmax=gdf_F["exposed_population"].max())
    # cmap_pop_exposed = mcolors.LinearSegmentedColormap.from_list("white_to_darkblue", ["#ffffff", "#67CBE4"])
    cmap_pop_exposed = mcolors.LinearSegmentedColormap.from_list(
    "white_blue_purple", ["#ffffff", "#67CBE4", "#3B1F8C"])
    cmap_change = plt.cm.Reds
    norm_cells_change = PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax(gdf_F["change_area_flooded"]))
    norm_pop_change = PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax(gdf_F["change_in_population"]))

    fig, axes = plt.subplots(1, 3, figsize=(11, 5), dpi=300, sharey=True, constrained_layout=True,
                             subplot_kw={"projection": ccrs.UTM(36, southern_hemisphere=True)})

    # Plot 1 - Factual exposed population 
    gdf_exposed_F = gdf_F.copy()
    gdf_exposed_F.loc[gdf_exposed_F["exposed_population"] == 0, "exposed_population"] = np.nan
    gdf_exposed_F.plot(column="exposed_population", cmap=cmap_pop_exposed, edgecolor="grey", 
                       norm=norm_pop_exposed, linewidth=0.2, ax=axes[0], legend=False, 
                       zorder=2, rasterized=True,
                       missing_kwds={"color": "none", "edgecolor": "none"})
    
    # Plot 2 - Climate change-induced flood extent
    gdf_exposed_F.loc[gdf_exposed_F["change_area_flooded"] == 0, "change_area_flooded"] = np.nan
    gdf_exposed_F.plot(column="change_area_flooded", cmap=cmap_change, edgecolor="grey", 
                       norm=norm_cells_change, linewidth=0.2, ax=axes[1], legend=False, 
                       zorder=2, rasterized=True,
                       missing_kwds={"color": "none", "edgecolor": "none"})
    
    # Plot 3 - Population change
    gdf_F.plot(column="change_in_population", cmap=cmap_change, norm=norm_pop_change, linewidth=0.1,
                  edgecolor="grey", ax=axes[2], zorder=2, missing_kwds={"color": "none", "edgecolor": "none"})
    
    setup_map_axes(axes, region_utm, background_utm, flood_extent,
                   subplot_labels=["(a)", "(b)", "(c)",],
                   titles=["Factual exposed population", "Change in flood extent", "Change in population"])
    
    for ax in axes:
        # Plot city and river locations and names
        ax.plot(34.862, -19.833, marker='o', color='black', markersize=3, markeredgecolor='white', transform=ccrs.PlateCarree(), zorder=5)
        text = ax.text(34.852, -19.89, "Beira", transform=ccrs.PlateCarree(), fontsize=8, color='black', zorder=5)
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        
        # Buzi River marker and label
        ax.plot(34.43, -19.89, marker='o', color='black', markersize=3, markeredgecolor='white', transform=ccrs.PlateCarree(), zorder=5)
        text2 = ax.text(34.44, -19.87, "Buzi River", transform=ccrs.PlateCarree(),
                        fontsize=8, color='black', zorder=5)
        text2.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        # Pungwe River marker and label
        ax.plot(34.543, -19.545, marker='o', color='black', markersize=3, markeredgecolor='white', transform=ccrs.PlateCarree(), zorder=5)
        text3 = ax.text(34.554, -19.52, "Pungwe River", transform=ccrs.PlateCarree(),
                        fontsize=8, color='black', zorder=5)
        text3.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

        # Plot background
        background_outside_box.plot(ax=ax, color='#E0E0E0', transform=ccrs.PlateCarree(), zorder=0)
        background_outside_box.boundary.plot(ax=ax, color="#818181", linewidth=0.2, 
                                             transform=ccrs.PlateCarree(), zorder=1)
    # Colour bars top row
    sm = ScalarMappable(cmap=cmap_pop_exposed, norm=norm_pop_exposed)
    sm._A = []
    cbar1 = fig.colorbar(sm, ax=axes[0], shrink=0.4)
    cbar1.set_label("Exposed population (×10³ people)", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)
    formatter = FuncFormatter(lambda x, _: f"{x/1000:.0f}")
    cbar1.ax.yaxis.set_major_formatter(formatter)

    sm = ScalarMappable(cmap=cmap_change, norm=norm_cells_change)
    sm._A = []
    cbar2 = fig.colorbar(sm, ax=axes[1], shrink=0.4)
    cbar2.set_label("Change in flood extent (km²)", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)
    formatter = FuncFormatter(lambda x, _: f"{x/1000000:.0f}")
    cbar2.ax.yaxis.set_major_formatter(formatter)
    
    sm = ScalarMappable(cmap=cmap_change, norm=norm_pop_change)
    sm._A = []
    cbar3 = fig.colorbar(sm, ax=axes[2], orientation="vertical", shrink=0.4)
    cbar3.set_label("Change in population (×10³ people)", fontsize=9)
    cbar3.ax.tick_params(labelsize=8)
    formatter = FuncFormatter(lambda x, _: f"{x/1000:.0f}")
    cbar3.ax.yaxis.set_major_formatter(formatter)

    total_exposed_F = gdf_exposed_F["exposed_population"].sum()
    exp_of_total_pop = total_exposed_F / gdf_F["total_population"].sum() * 100
    axes[0].text(0.98, 0.98, f"{round(total_exposed_F, -3):,.0f} people", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round",pad=0.25, fc="white", ec="none", alpha=0.8))
    axes[0].text(0.98, 0.92, f"{exp_of_total_pop:.0f}% of total population", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round",pad=0.25, fc="white", ec="none", alpha=0.8))
    attr_flood_extent = gdf_F["attr_flood_extent"].iloc[0]
    axes[1].text(0.98, 0.98, f"{attr_flood_extent:.0f}% increase", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round",pad=0.25, fc="white", ec="none", alpha=0.8))
    gdf_F["tot_pop_change"] = (gdf_F["change_in_population"].sum()) / (gdf_CF_pop["total_population"].sum()) * 100
    axes[2].text(0.98, 0.98, f"{gdf_F['tot_pop_change'].iloc[0]:.0f}% increase", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round",pad=0.25, fc="white", ec="none", alpha=0.8))
    
    # fig.savefig("../figures/f02.png", dpi=300, bbox_inches='tight')
    # fig.savefig("../figures/f02.pdf", dpi=300, bbox_inches='tight')
    # fig.savefig("../figures/f02.jpeg", dpi=300, bbox_inches='tight')

    return fig


fig = plot_factual_and_driver_changes_extent(coarse_gdfs[2020]["F"], coarse_gdfs[2020]["CF"], 
                                             coarse_gdfs[1975]["F"], region_utm, background_utm, flood_extent)
plt.show()



#%%
# FIGURE 3: Plotting attributable exposed population per flood depth category
print("Plotting attributable exposed population (three drivers)")
# --- Compute differences ---
gdf_factual = coarse_gdfs[2020]["F"]

datasets = {"Climate change": coarse_gdfs[2020]["CF"].copy(),
            "Population change": coarse_gdfs[1975]["F"].copy(),
            "Climate & population change": coarse_gdfs[1975]["CF"].copy()}

for gdf in datasets.values():
    gdf["diff"] = (gdf_factual["exposed_population"] - gdf["exposed_population"])

total_factual = gdf_factual["exposed_population"].sum()

# --- Figure ---
fig, axes = plt.subplots(1, 3, figsize=(11, 5), dpi=300, constrained_layout=True,
                         subplot_kw={"projection": ccrs.UTM(36, southern_hemisphere=True)})

# Shared normalization
vmax = max(ds["diff"].max() for _, ds in datasets.items())
vmin = min(ds["diff"].min() for _, ds in datasets.items())
norm_diff = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
cmap = plt.cm.Reds 
cmap.set_bad("white")   # for masked zeros if needed

subplot_labels = ["(a)", "(b)", "(c)"]
titles = list(datasets.keys())

setup_map_axes(axes, region_utm=region_utm, background_utm=background_utm, flood_extent=flood_extent,
               subplot_labels=subplot_labels, titles=titles, show_left_labels_only=True,
               background_outside_box=background_outside_box)

# --- Plot loop ---
for ax, (title, gdf) in zip(axes, datasets.items()):
    gdf_plot = gdf.copy()
    gdf_plot.loc[gdf_plot["diff"] == 0, "diff"] = np.nan
    gdf_plot.plot(column="diff", cmap=cmap, norm=norm_diff, edgecolor="grey", linewidth=0.2, ax=ax,
                  legend=False, zorder=2, rasterized=True, missing_kwds={"color": "white"})
    
    # Plot the boundary of the Beira district and reference locations
    beira_utm.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.7, zorder=3)
    add_reference_locations(ax)

    # Annotate the absolute and relative change in exposed population
    rel_change = gdf["diff"].sum() / total_factual * 100
    ax.text(0.98, 0.98, f"{round(gdf['diff'].sum(), -3):,.0f} people\n({rel_change:.0f}%)", 
            transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

# --- Shared colorbar ---
fmt = FuncFormatter(lambda x, pos: f"{int(x):,}")
sm = ScalarMappable(cmap=cmap, norm=norm_diff)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.6, pad=0.02)
cbar.set_label("Attributable exposed population (# people)", fontsize=10)
cbar.ax.tick_params(labelsize=9)
cbar.ax.yaxis.set_major_formatter(fmt)

# fig.savefig("../figures/f03.png", dpi=300, bbox_inches='tight')
# fig.savefig("../figures/f03.pdf", dpi=300, bbox_inches='tight')
# fig.savefig("../figures/f03.jpeg", dpi=300, bbox_inches='tight')


#%%
# --- Stats of exposed population change in Beira district ---
beira_scenarios = {"CF_climate": coarse_gdfs[2020]["CF"],
                   "CF_population": coarse_gdfs[1975]["F"],
                   "CF_climate_population": coarse_gdfs[1975]["CF"]}

beira_cells = {}
for name, gdf in beira_scenarios.items():
    beira_cells[name] = gpd.overlay(gdf, beira_utm, how="intersection")

beira_cells_F = gpd.overlay(coarse_gdfs[2020]["F"], beira_utm, how="intersection")

for gdf in beira_cells.values():
    gdf["diff"] = None

for name, gdf in beira_cells.items():
    print(f"Total exposed population change in Beira District ({name}): "
          f"{round(gdf['diff'].sum(), -3):,.0f}")

pop_2020 = beira_cells_F["total_population"].sum()
pop_1975 = beira_cells["CF_population"]["total_population"].sum()

growth_abs = pop_2020 - pop_1975
growth_pct = growth_abs / pop_1975 * 100

print(f"Total population in Beira in 2020: {round(pop_2020, -3):,.0f}")
print(f"Total population in Beira in 1975: {round(pop_1975, -3):,.0f}")
print(f"Population growth in Beira: " f"{round(growth_abs, -3):,.0f} people ({growth_pct:.1f} %)")


#%% 
# FIGURE 4 & Table S06: Flood depth category change and bar plot of attributable fraction per category
# --- Classify depth raster cells directly ---
def depth_category_array(hmax):
    cat = np.full(hmax.shape, 'None', dtype=object)
    cat[(hmax >= 0.15) & (hmax < 0.5)]     = 'Low'
    cat[(hmax >= 0.5) & (hmax <= 1.5)]    = 'Mod-high'
    cat[hmax > 1.5]                       = 'Very high'
    return cat

# --- Compute transition categories ---
def compute_transition(cat_cf, cat_f, mask=None):
    if mask is None:
        mask = np.ones(cat_cf.shape, dtype=bool)

    cf = np.where(mask, cat_cf, "None")
    f  = np.where(mask, cat_f, "None")

    transition = np.full(cat_cf.shape, "None", dtype=object)
    transition[(cf == f) & (f != "None")] = "No change"
    transition[(cf == "None") & (f == "Low")] = "None → Low"
    transition[(cf == "None") & (f == "Mod-high")] = "None → Mod-high"
    transition[(cf == "None") & (f == "Very high")] = "None → Very high"
    transition[(cf == "Low") & (f == "Mod-high")] = "Low → Mod-high"
    transition[(cf == "Low") & (f == "Very high")] = "Low → Very high"
    transition[(cf == "Mod-high") & (f == "Very high")] = "Mod-high → Very high"

    return transition

# --- Function to get transition counts for a given transition category ---
def transition_counts_from_array(transition, all_transitions):
    return (pd.Series(transition.ravel()).value_counts().reindex(all_transitions, fill_value=0))

# --- Convert arrays to RGBA images for plotting ---
def cat_array_to_rgba(cat_array, color_dict):
    rgba = np.zeros((*cat_array.shape, 4), dtype=np.float32)
    for label, hex_color in color_dict.items():
        rgb = mcolors.to_rgba(hex_color)
        mask = cat_array == label
        rgba[mask] = rgb
    return rgba


# --- Color maps ---
depth_colors = {'Low': '#9ECAE1', 'Mod-high': '#3182BD', 'Very high':'#08306B'}
all_transitions = ['No change', 'None → Low', 'None → Mod-high',
                   'None → Very high', 'Low → Mod-high', 
                   'Low → Very high', 'Mod-high → Very high']
transition_codes = {'No change': 0, 'None → Low': 1, 'None → Mod-high': 2,
                    'None → Very high': 3, 'Low → Mod-high': 4,
                    'Low → Very high': 5, 'Mod-high → Very high': 6}
transition_colors = {'No change':       '#CCCCCC',
                     'None → Low':      '#fd8d3c',
                     'None → Mod-high': '#e6550d',
                    #  'None → Very high':'#7F2704',
                     'Low → Mod-high':  '#A1D99B',
                    #  'Low → Very high': '#238B45',
                     'Mod-high → Very high': '#00441B'}
# Transition colors with transparent for NaNs
cmap_categories = ListedColormap(list(transition_colors.values()))
cmap_categories.set_bad(color="none")


# Classify depth categories for both scenarios and calculate transition
cat_F  = depth_category_array(hmax_F)
cat_CF = depth_category_array(hmax_CF)
mask_exposed = (exposed_rasters[2020]["F"] > 0) | (exposed_rasters[2020]["CF"] > 0)

transition = compute_transition(cat_CF, cat_F)
transition_counts = transition_counts_from_array(transition, all_transitions)
transition_counts.to_csv("../results/Table_S06.csv")
transition_exp = compute_transition(cat_CF, cat_F, mask=mask_exposed)

rgba_F          = cat_array_to_rgba(cat_F, depth_colors)
rgba_transition = cat_array_to_rgba(transition, transition_colors)
rgba_transition_exp = cat_array_to_rgba(transition_exp, transition_colors)

# Create a DataFrame for population counts by transition and urban/rural
df = pd.DataFrame({"transition": transition_exp.ravel(),
                   "population": exposed_rasters[2020]["F"].ravel(),
                   "urban": urban_mask.ravel()})
df = df.query("transition != 'None' and population > 0")
df["area_type"] = np.where(df["urban"] == 1, "Urban", "Rural")
table = (df.groupby(["transition", "area_type"])["population"].sum().unstack(fill_value=0))
table_cat_change_settlement = table.reindex(all_transitions, fill_value=0)

transition_numeric = np.full(transition_exp.shape, np.nan, dtype=np.float32)
for k, v in transition_codes.items():
    transition_numeric[transition_exp == k] = v

print("Change in flood depth category among exposed population (by settlement type):")
print(table_cat_change_settlement)
table_cat_change_settlement.to_csv("../results/transition_by_settlement_type.csv")


#%%
# Plotting settings for Figure 4
# To transfrom coordinates from WSG84 to UTM 36s
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32736", always_xy=True)

# for zoom in
beira_extent = [34.82, 34.925, -19.86, -19.77]
buzi_extent  = [34.57, 34.62, -19.895, -19.87]
beira_extent_utm = extent_to_utm(beira_extent)
buzi_extent_utm  = extent_to_utm(buzi_extent)

def add_beira_context(
    ax,
    background,
    urban_gdf=None,
    mask_box=(34.8, -20.3, 35.3, -19.9),
):
    """
    Adds:
    - clipped background
    - Beira marker + label
    - river markers + labels
    - optional urban overlay
    """

    # --- Clip background ---
    background_out = background[~background.intersects(box(*mask_box))]

    background_out.plot(
        ax=ax,
        color="#E0E0E0",
        transform=ccrs.PlateCarree(),
        zorder=0,
    )

    background_out.boundary.plot(
        ax=ax,
        color="#818181",
        linewidth=0.2,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    # --- Static points (Beira + rivers) ---
    features = [
        (34.862, -19.833, "Beira", (34.852, -19.89)),
        (34.43,  -19.89,  "Buzi River", (34.29, -19.882)),
        (34.543, -19.545, "Pungwe River", (34.35, -19.538)),
    ]

    for lon, lat, label, (tx, ty) in features:
        ax.plot(
            lon, lat, "o",
            color="black",
            markersize=3,
            markeredgecolor="white",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        t = ax.text(
            tx, ty, label,
            transform=ccrs.PlateCarree(),
            fontsize=8,
            zorder=5,
        )

        t.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal()
        ])

    # --- Optional urban overlay ---
    if urban_gdf is not None:
        urban_gdf.boundary.plot(
            ax=ax,
            edgecolor="#a6761d",
            linewidth=0.8,
            zorder=2,
        )


# ========================= FIGURE 4 =========================

fig = plt.figure(figsize=(10, 10), dpi=300, constrained_layout=True)

gs = GridSpec(2, 4, figure=fig, height_ratios=[2, 1.8], hspace=0.05, wspace=0.05)

ax0 = fig.add_subplot(gs[0, 0:2], projection=ccrs.UTM(36, southern_hemisphere=True))
ax1 = fig.add_subplot(gs[0, 2:4], projection=ccrs.UTM(36, southern_hemisphere=True))

gs_bottom = GridSpecFromSubplotSpec(1, 10, subplot_spec=gs[1, :])
ax2 = fig.add_subplot(gs_bottom[0, 2:8])

axes = [ax0, ax1, ax2]

# --- Raster panels ---
ax0.imshow(
    rgba_F,
    extent=flood_extent,
    origin="lower",
    transform=ccrs.UTM(36, southern_hemisphere=True),
    zorder=2,
)

ax1.imshow(rgba_transition, extent=flood_extent, origin="lower",
           transform=ccrs.UTM(36, southern_hemisphere=True), zorder=2)

# --- Context (Beira, rivers, background, optional urban) ---
for ax in axes[:2]:
    add_beira_context(ax, background, urban_gdf=gdf_urban)

setup_map_axes(axes[:2], region_utm, background_utm, flood_extent, subplot_labels=["(a)", "(b)"],
               titles=["", ""], axis_labelsize=10, subplot_labelsize=11)

# ========================= PANEL (c): BAR PLOT =========================
bar_width = 0.2
x_pos = np.arange(3)
labels = ["Low\n(0.15–0.5 m)", "Mod-high\n(0.5–1.5 m)", "Very high\n(>1.5 m)"]

ax2.bar(x_pos - bar_width, data_attr[:, 0], width=bar_width,
        label=f"Climate change ({int(np.round(perct_attr_clim))} %)", color=colours[1])
ax2.bar(x_pos, data_attr[:, 1], width=bar_width,
        label=f"Population change ({int(np.round(perct_attr_pop))} %)", color=colours[2])
ax2.bar(x_pos + bar_width, data_attr[:, 2], width=bar_width, 
        label=f"Climate + population ({int(np.round(perct_attr_clim_pop))} %)", color=colours[3])

# --- Formatting ---
ax2.axhline(0, linestyle="-", color="black", alpha=0.7)
ax2.set_axisbelow(True)
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
ax2.set_xlabel("Flood depth category", fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontdict={"fontweight": "bold", 'color': '#5C5C5C', "fontsize": 9})
ax2.set_ylabel("Attributable exposed population (%)", fontsize=11)
ax2.tick_params(axis="y", labelsize=9)

# --- Background shading ---
ymin, ymax = ax2.get_ylim()
xmin, xmax = ax2.get_xlim()
xticks = ax2.get_xticks()

boundaries = ([xmin] +                                                           
        [(xticks[j] + xticks[j+1]) / 2 for j in range(len(xticks)-1)] +  
        [xticks[-1] + (xticks[-1] - xticks[-2]) / 2])

shade_colors = ["#d9d9d9", "#b3b3b3", "#808080"]
ax2.set_ylim(ax2.get_ylim())
ax2.set_xlim(ax2.get_xlim())  
for j in range(len(xticks)):
    ax2.fill_betweenx([ymin, ymax], boundaries[j], boundaries[j + 1], color=shade_colors[j],
                      alpha=0.3, zorder=0)
  
# --- Legend and label ---
ax2.legend(fontsize=9, loc="upper right")
ax2.text(0, 1.02, "(c)", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")

# ========================= SAVE =========================
# fig.savefig("figures/f04.png", dpi=300, bbox_inches="tight")
# fig.savefig("figures/f04.pdf", dpi=300, bbox_inches="tight")
# fig.savefig("figures/f04.jpeg", dpi=300, bbox_inches="tight")

plt.show()


#%% #############################################################################################
#################################################################################################
#################################### SUPPLEMENTARY FIGURES ###################################### 
#################################################################################################
#################################################################################################
# ------------------------------------------------------------ #
# SUMMARY TABLE OF POPULATION EXPOSED PER FLOOD DEPTH CATEGORY #
# ------------------------------------------------------------ #
# TABLE S5
# Depth bins
depth_bins = {"0.15-0.5 m": (0.15, 0.5),
              "0.5-1.5 m": (0.5, 1.5), 
              ">1.5 m": (1.5, np.inf),
              "Total": (0, np.inf)}

# Scenarios 
scenarios = {"F": (exposed_gdfs[2020]["F"], 2020),
             "CF_clim": (exposed_gdfs[2020]["CF"], 2020),
             "CF_pop": (exposed_gdfs[1975]["F"], 1975),
             "CF_clim_pop": (exposed_gdfs[1975]["CF"], 1975)}

total_population = {2020: np.nansum(pop_data[2020]["population"]),
                    1975: np.nansum(pop_data[1975]["population"])}

data = {}
for scenario, (gdf, year) in scenarios.items():
    data[scenario] = {}
    for depth_label, (dmin, dmax) in depth_bins.items():
        mask = (gdf["flood_depth"] > dmin) & (gdf["flood_depth"] <= dmax)
        exposed = np.nansum(gdf.loc[mask, "population"])
        data[scenario][depth_label] = {"total_pop": total_population[year],
                                       "exposed_pop": exposed}

for scenario in scenarios:
    if scenario == "F":
        continue        
    for depth_label in depth_bins:
        F_val = data["F"][depth_label]["exposed_pop"]
        CF_val = data[scenario][depth_label]["exposed_pop"]
        data[scenario][depth_label]["abs_change"] = F_val - CF_val
        data[scenario][depth_label]["attributable"] = (F_val - CF_val)/F_val * 100

for depth_label in depth_bins:
    data["F"][depth_label]["abs_change"] = 0
    data["F"][depth_label]["attributable"] = 0

rows = []
for scenario in scenarios:
    row = {}    
    for depth_label in depth_bins:
        for metric in ["total_pop", "exposed_pop", "abs_change", "attributable"]:
            col = (depth_label, metric)
            row[col] = data[scenario][depth_label].get(metric, np.nan)
    rows.append(pd.Series(row, name=scenario))

df_final = pd.DataFrame(rows)
df_final.columns = pd.MultiIndex.from_tuples(df_final.columns)
print(df_final)

# Export
df_change_flood_category = df_final.copy()
df_change_flood_category.columns = [f"{depth}_{metric}" for depth, metric in df_change_flood_category.columns]
# df_change_flood_category.to_csv("../results/Table_S05.csv")



 #%%
# FIGURE FS2 — Overview of factual changes in flood hazard and population exposure
def plot_supp_factual_changes_overview(hmax_F_da, gdf_pop_2019_exposed_F_coarse, hmax_diff, 
                                       gdf_pop_1975_exposed_F_coarse, gdf_pop_1975_exposed_CF_coarse,
                                       region_utm, background_utm, flood_extent):
    # Data preparation for plotting
    gdf_2019 = gdf_pop_2019_exposed_F_coarse.copy()
    gdf_2019.loc[gdf_2019["total_population"] == 0] = np.nan
    gdf_1975 = gdf_pop_1975_exposed_F_coarse.copy()
    gdf_1975.loc[gdf_1975["total_population"] == 0] = np.nan
    gdf_2019["change_in_population"] = gdf_2019["total_population"] - gdf_1975["total_population"]

    # colour maps and norms
    norm_pop = PowerNorm(gamma=0.5, vmin=0, vmax=gdf_2019["total_population"].max())
    cmap_pop = mcolors.LinearSegmentedColormap.from_list("white_to_orange", ["#ffffff", "#FC6F37"])
    cmap_change = plt.cm.Reds
    cmap_change.set_bad((0, 0, 0, 0))  # fully transparent
    norm_pop_change = PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax(gdf_2019["change_in_population"]))

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5), dpi=300, sharex=True, sharey=True, constrained_layout=True,
                             subplot_kw={"projection": ccrs.UTM(36, southern_hemisphere=True)})

    # Plot 1 - Factual fooding
    im_1 = axes[0,0].imshow(hmax_F_da.values, cmap="viridis", extent=flood_extent, origin="lower", vmin=0.15, vmax=3.5, zorder=2)
    
    # Plot 2 - Climate change
    im_2 = axes[0,1].imshow(hmax_diff, cmap=cmap_change, extent=flood_extent, origin="lower", vmin=0, vmax=0.5, zorder=2)

    # Plot 3 - Factual population
    gdf_2019.plot(column="total_population", cmap=cmap_pop, norm=norm_pop, linewidth=0.1,
                  edgecolor="grey", ax=axes[1,0], zorder=2, missing_kwds={"color": "none", "edgecolor": "none"})
    
    # Plot 4 - Population change
    gdf_2019.plot(column="change_in_population", cmap=cmap_change, norm=norm_pop_change, linewidth=0.1,
                  edgecolor="grey", ax=axes[1,1], zorder=2, missing_kwds={"color": "none", "edgecolor": "none"})

    setup_map_axes(axes, region_utm, background_utm, flood_extent,
                   subplot_labels=["(a)", "(b)", "(c)", "(d)"],
                   titles=["Factual flooding", "Climate change", "Factual population", "Population change"])

    # Colour bars top row
    cbar1 = fig.colorbar(im_1, ax=axes[0,0], shrink=0.8)
    cbar1.set_label("Flood depth (m)")
    cbar1.set_ticks([0.15,0.5,1,1.5,2,2.5,3,3.5])
    cbar1.set_ticklabels(["0.15","0.5","1","1.5","2","2.5","3","3.5"])

    plt.colorbar(im_2, ax=axes[0,1], shrink=0.8).set_label("Attributable flood depth (m)")

    # Colour bars for the bottom row
    sm = ScalarMappable(cmap=cmap_pop, norm=norm_pop)
    sm._A = []
    cbar3 = fig.colorbar(sm, ax=axes[1,0], shrink=0.8)
    cbar3.set_label("Total population (×10³ people)", fontsize=10)
    cbar3.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))

    sm = ScalarMappable(cmap=cmap_change, norm=norm_pop_change)
    sm.set_array([])
    cbar4 = fig.colorbar(sm, ax=axes[1,1], shrink=0.8)
    cbar4.set_label("Change in population (×10³ people)", fontsize=10)
    cbar4.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))

    # fig.savefig("../figures/fS02.png", dpi=300, bbox_inches="tight")
    # fig.savefig("../figures/fS02.pdf", dpi=300, bbox_inches="tight")
    # fig.savefig("../figures/fS02.jpeg", dpi=300, bbox_inches="tight")

    return fig


fig = plot_supp_factual_changes_overview(hmax_F_da, coarse_gdfs[2020]['F'], hmax_diff, coarse_gdfs[1975]['F'], 
                                         coarse_gdfs[1975]['CF'], region_utm, background_utm, flood_extent)
plt.show()


#%%
# Figure S3 — Absolute numbers of attribution of different drivers per flood depth category
fig, ax = plt.subplots(1, 1, figsize=(7,5), sharex=True, dpi=300, constrained_layout=True)

bar_width = 0.25
x_pos = np.arange(3)  # Low, Medium, High
labels = ["Low \n(0.15–0.5 m)", "Mod-high \n(0.5–1.5 m)", "Very high \n(>1.5 m)"]
boundaries = [-0.5, 0.5, 1.5, 2.5]
shade_colors = ["#d9d9d9", "#b3b3b3", "#808080"]

ymin, ymax = 0, max(data_abs_diff.max(), 1) * 1.1

for j in range(len(shade_colors)):
    ax.fill_betweenx([ymin, ymax], boundaries[j], boundaries[j+1], color=shade_colors[j], alpha=0.3, zorder=0)
    
ax.bar(x_pos - bar_width, data_abs_diff[:,0], width=bar_width, 
       label=f"Climate change ({int(np.round(perct_attr_clim))} %)", 
       color=colours[1])
ax.bar(x_pos, data_abs_diff[:,1], width=bar_width, 
       label=f"Population change ({int(np.round(perct_attr_pop))} %)", 
       color=colours[2])
ax.bar(x_pos + bar_width, data_abs_diff[:,2], width=bar_width, 
       label=f"Climate change & \nPopulation change ({int(np.round(perct_attr_clim_pop))} %)", 
       color=colours[3])

ax.set_axisbelow(True)
ax.grid(True, axis='y', linestyle="--", alpha=0.5)
ax.set_xlabel("Flood depth", fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontdict={'fontweight': 'bold', 'color': '#5C5C5C'})
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))
ax.set_ylim(0, ymax)
ax.set_xlim(-0.5, 2.5)

ax.set_ylabel("Absolute change in exposed population (×10³ people)", fontsize=11)
ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1, 1))


# fig.savefig("../figures/fS03.png", dpi=300, bbox_inches='tight')
# fig.savefig("../figures/fS03.pdf", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()


# with arrows indicating peak differences
def match_peaks_by_x(x, peaksA, peaksB):
    """
    Match peaks between two scenarios by closest x position.
    Returns list of tuples: (peakA_idx, peakB_idx)
    where either can be None if unmatched.
    """
    xsA = x[peaksA]
    xsB = x[peaksB]

    matched = []
    used_B = set()

    # Match each A peak to nearest B peak
    for iA, xA in zip(peaksA, xsA):
        if len(xsB) > 0:
            # find closest B peak not yet used
            diffs = [(abs(xA - xB), iB) for xB, iB in zip(xsB, peaksB) if iB not in used_B]
            if diffs:
                _, bestB = min(diffs, key=lambda t: t[0])
                used_B.add(bestB)
                matched.append((iA, bestB))
            else:
                matched.append((iA, None))
        else:
            matched.append((iA, None))

    # Add B peaks with no match in A
    for iB in peaksB:
        if iB not in used_B:
            matched.append((None, iB))

    return matched

# Draw arrows between matched peaks
def draw_peak_arrows(x, yA, yB, peaksA, peaksB, color, label_prefix, label_offsets, arrow_offsets,
                     min_diff=0.01):

    pairs = match_peaks_by_x(x, peaksA, peaksB)

    for i, (iA, iB) in enumerate(pairs):

        # CASE 1: Peak exists in both lines --------------------------------------
        if iA is not None and iB is not None:
            xA, yA_val = x[iA], yA[iA]
            xB, yB_val = x[iB], yB[iB]

        # CASE 2: Peak in A but not in B -----------------------------------------
        elif iA is not None and iB is None:
            xA = x[iA]
            yA_val = yA[iA]
            xB = xA  # vertical arrow
            yB_val = yB[iA]  # value on other line at same x

        # CASE 3: Peak in B but not in A -----------------------------------------
        elif iB is not None and iA is None:
            xB = x[iB]
            yB_val = yB[iB]
            xA = xB  # vertical arrow
            yA_val = yA[iB]  # value on other line at same x

        # compute difference
        diff = abs(yA_val - yB_val)

        # FILTER 1 — small differences
        if diff < min_diff:
            continue

        # vertical arrow if xA == xB (peak ↔ no peak)
        offset_x_arrow, offset_y_arrow = arrow_offsets.get(i, (0.0, 0.0))
        plt.annotate(
            "",
            xy=(xB+offset_x_arrow, yB_val+offset_y_arrow),
            xytext=(xA, yA_val),
            arrowprops=dict(arrowstyle="->", lw=1.4, color=color))

        # add label
        offset_x, offset_y = label_offsets.get(i, (0.03, 0.0)) 
        xm = xA + offset_x  # midpoint in x (same if vertical)
        ym = (yA_val + yB_val)/2 + offset_y
        plt.text(xm, ym, f"{label_prefix}",
                 ha="left", va="center", fontsize=9, color=color, fontweight="bold",
                 bbox=dict(boxstyle="round",pad=0.15, facecolor="lightgrey", 
                           edgecolor="none", alpha=0.5))

#%%
# Figure S4 — Plotting absolute change in exposed population per water depth
fig, ax = plt.subplots(figsize=(8,5), dpi=300)

x = bin_centers
y_F = depth_dist_fine["2020_F"].values
y_CF_clim = depth_dist_fine["2020_CF"].values
y_CF_pop = depth_dist_fine["1975_F"].values
y_CF_clim_pop = depth_dist_fine["1975_CF"].values

ymax = max(y_F.max(), y_CF_clim.max(), y_CF_pop.max(), y_CF_clim_pop.max()) * 1.05

ax.fill_between(x_bg[low_mask_bg], 0, ymax, color="#d9d9d9", alpha=0.3)
ax.fill_between(x_bg[mid_mask_bg], 0, ymax, color="#b3b3b3", alpha=0.3)
ax.fill_between(x_bg[high_mask_bg], 0, ymax, color="#808080", alpha=0.3)

ax.plot(x, y_F, label=f"Factual ({np.round(np.nansum(exposed_rasters[2020]['F']), -3):,.0f} people)", color=colours[0], linewidth=2)
ax.plot(x, y_CF_clim, label=f"Counterfactual Climate ({np.round(np.nansum(exposed_rasters[2020]['CF']), -3):,.0f} people)", color=colours[1], linewidth=1)
ax.plot(x, y_CF_pop, label=f"Counterfactual Population ({np.round(np.nansum(exposed_rasters[1975]['F']), -3):,.0f} people)", color=colours[2], linewidth=1)
ax.plot(x, y_CF_clim_pop, label=f"Counterfactual Climate & Population ({np.round(np.nansum(exposed_rasters[1975]['CF']), -3):,.0f} people)", color=colours[3], linewidth=1)

# Find peaks and sort by height
peaks_F, props_F = find_peaks(y_F, prominence=0.02, distance=5) 
sorted_peaks_F = peaks_F[np.argsort(props_F["prominences"])[::-1]]
peaks_CF_clim, props_CF_clim = find_peaks(y_CF_clim, prominence=0.02, distance=5)
sorted_peaks_CF_clim = peaks_CF_clim[np.argsort(props_CF_clim["prominences"])[::-1]]
peaks_CF_pop, props_CF_pop = find_peaks(y_CF_pop, prominence=0.02, distance=10)
sorted_peaks_CF_pop = peaks_CF_pop[np.argsort(props_CF_pop["prominences"])[::-1]]
# find second peak for CF_pop manually due to slowrise
mask = (x >= 1.0) & (x <= 1.5) # peak is located between 1 and 1.5 m flood depth
if np.any(mask):
    idx_peak_2nd_CF_pop = np.argmax(y_CF_pop[mask])
    idx_peak_2nd_CF_pop = np.where(mask)[0][idx_peak_2nd_CF_pop]

# peak annotations 
if len(sorted_peaks_F) > 1:
    second_peak_x = x[sorted_peaks_F[1]]
    plt.annotate(f"{x[sorted_peaks_F[0]]:.2f} m",
                 xy=(x[sorted_peaks_F[0]]+0.02, y_F[sorted_peaks_F[0]]),
                 xytext=(x[sorted_peaks_F[0]]+0.15, y_F[sorted_peaks_F[0]]-0.05), color=colours[0], fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=colours[0], linestyle='--'))
    plt.annotate(f"{x[sorted_peaks_F[1]]:.2f} m", 
                 xy=(x[sorted_peaks_F[1]]+0.02, y_F[sorted_peaks_F[1]]),
                 xytext=(x[sorted_peaks_F[1]]+0.15, y_F[sorted_peaks_F[1]]), color=colours[0], fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=colours[0], linestyle='--'))

if len(sorted_peaks_CF_clim) > 1:
    second_peak_x = x[sorted_peaks_CF_clim[1]]
for i, idx in enumerate(sorted_peaks_CF_clim[:2]): 
    plt.annotate(f"{x[idx]:.2f} m",
                 xy=(x[idx]+0.02, y_CF_clim[idx]),
                 xytext=(x[idx]+0.15, y_CF_clim[idx]+0.05), color=colours[1], fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=colours[1], linestyle='--'))

for i, idx in enumerate(sorted_peaks_CF_pop[:1]): 
    plt.annotate(f"{x[idx]:.2f} m",
                 xy=(x[idx]+0.01, y_CF_pop[idx]),
                 xytext=(x[idx]-0.02, y_CF_pop[idx]+300), color=colours[2], fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=colours[2], linestyle='--'))
    
ax.text(0.32, ymax*0.03, "Low", ha="center", fontweight="bold", color="#5C5C5C", fontsize=10)
ax.text(1.0, ymax*0.03, "Mod-high", ha="center", fontweight="bold", color="#5C5C5C", fontsize=10)
ax.text(2.5, ymax*0.03, "Very high", ha="center", fontweight="bold", color="#5C5C5C", fontsize=10)

ax.set_xlabel("Flood depth (m)")
ax.set_ylabel("Exposed population (# people)")
ax.set_xlim(0.15, 3.5)
ax.set_ylim(0, ymax)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

# fig.savefig("../figures/fS04.png", dpi=300, bbox_inches='tight')
# fig.savefig("../figures/fS04.pdf", dpi=300, bbox_inches='tight')

# %%
# Figure S5 — Compare exposure change in the case of uniform growth per water depth
fig, ax = plt.subplots(figsize=(8,5), dpi=300)

x = bin_centers
y_F = depth_dist_fine["2020_F"].values
y_CF_uni = depth_dist_fine["2020_uniform_F"].values

ymax = max(y_F.max(), y_CF_uni.max()) * 1.05

ax.fill_between(x_bg[low_mask_bg], 0, ymax, color="#d9d9d9", alpha=0.3)
ax.fill_between(x_bg[mid_mask_bg], 0, ymax, color="#b3b3b3", alpha=0.3)
ax.fill_between(x_bg[high_mask_bg], 0, ymax, color="#808080", alpha=0.3)

ax.plot(x, y_F, label=f"Factual ({np.round(np.nansum(exposed_rasters[2020]['F']), -3):,.0f} people)", color=colours[0], linewidth=2)
ax.plot(x, y_CF_uni, label=f"Uniform growth ({np.round(np.nansum(uniform_exposed_rasters['F']), -3):,.0f} people)", color="#1B4332", linewidth=1)
    
ax.text(0.32, ymax*0.065, "Low", ha="center", fontweight="bold", color="#5C5C5C", fontsize=10)
ax.text(1.1, ymax*0.065, "Mod-high", ha="center", fontweight="bold", color="#5C5C5C", fontsize=10)
ax.text(2.5, ymax*0.065, "Very high", ha="center", fontweight="bold", color="#5C5C5C", fontsize=10)

ax.set_xlabel("Flood depth (m)")
ax.set_ylabel("Exposed population (# people)")
ax.set_xlim(0.15, 3.5)
ax.set_ylim(0, ymax)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

# fig.savefig("../figures/fS05.png", dpi=300, bbox_inches='tight')
# fig.savefig("../figures/fS05.pdf", dpi=300, bbox_inches='tight')

