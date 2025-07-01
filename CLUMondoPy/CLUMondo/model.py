import os
from osgeo import gdal
import numpy as np
import pandas as pd
import random
import rasterio
from typing import List, Optional, Union

from io_utils import writeArray2GeoTIFF, check_no_data_value
from logging_utils import create_timestamped_subfolder, log_metadata, log_initial_data
from neighbourhood import calc_neigh
from transitions import calc_change
from demand import comp_demand
from age import calc_age, autonomous_change

def clumondo_dynamic(land_array: np.ndarray,
                     suit_array: np.ndarray,
                     region_array: np.ndarray,
                     neigh_weights: np.ndarray,
                     start_year: int,
                     end_year: int,
                     demand: np.ndarray,
                     dem_weights: np.ndarray,
                     lus_conv: np.ndarray,
                     lus_matrix_path: str,
                     allow: np.ndarray,
                     conv_res: np.ndarray,
                     max_diff_allow: float,
                     totdiff_allow: float,
                     max_iter: int,
                     out_dir: str,
                     crs: str,
                     dtype: str,
                     ref_raster_path: str,
                     change_years: List[int],
                     change_paths: List[str],
                     metadata: List[str],
                     age_array: Optional[np.ndarray] = None,
                     zonal_array: Optional[np.ndarray] = None,
                     preference_array: Optional[np.ndarray] = None,
                     preference_weights: Optional[np.ndarray] = None,
                     width_neigh: int = 1,
                     demand_max: float = 3,
                     demand_setback: float = 0.5,
                     no_data_out: int = 9999,
                     out_year: Optional[Union[int, List[int]]] = None,
                     no_data_value: int = -9999) -> None:
    """
    Perform Clumondo model simulation.

    Parameters:
        land_array (numpy.ndarray): Array representing the initial land cover.
        suit_array (numpy.ndarray): Array representing regional suitability values.
        region_array (numpy.ndarray): Array representing regions where land cover change may be restricted.
        neigh_weights (numpy.ndarray): Array of weights for neighborhood influence on land cover change.
        start_year (int): Start year of the simulation.
        end_year (int): End year of the simulation.
        demand (list of numpy.ndarray): List of arrays representing land use service demands for each year.
        dem_weights (numpy.ndarray): Array of weights for land use services demands.
        lus_conv (numpy.ndarray): Array representing conversion factors for land use services demands.
        lus_matrix_path (str): Array representing land use service matrix OR path to dynamic lus_matrix files.
        allow (numpy.ndarray): Array representing allowed land use changes.
        max_diff_allow (float): Maximum allowed difference in land cover change.
        totdiff_allow (float): Maximum allowed total difference in land cover change.
        max_iter (int): Maximum number of iterations for each year.
        out_dir (str): Output directory path where results are stored.
        #add crs, dtype and no_data_out
        ref_raster_path (str): Path to the reference raster file.
        change_years (list of int): Years where changes in suitability (`suit_array`) occur.
        change_paths (list of str): Paths to the suitability change raster files.
        age_array (numpy.ndarray, optional): Array representing the age of land cover. Defaults to None.
        no_data_value (int, optional): Value representing no data. Defaults to -9999.
    """
    # Information from reference raster to write rasters in the function
    raster_ds = gdal.Open(ref_raster_path)
    # Extract information from reference raster
    cols = raster_ds.RasterXSize
    rows = raster_ds.RasterYSize
    cell_res = int(raster_ds.GetGeoTransform()[1])
    x_origin = int(raster_ds.GetGeoTransform()[0])
    y_origin = int(raster_ds.GetGeoTransform()[3] - (cell_res * rows))
    ras_info = [cols, rows, x_origin, y_origin, cell_res]
    ras_info.append(no_data_value)

    # Update region array with no data values from suitability layer
    region_array[suit_array[0]==no_data_value] = 1

    if out_year is not None:
        if isinstance(out_year, int):
            out_year = [out_year]
        elif not isinstance(out_year, list):
            raise ValueError("Parameter 'out_year' must be an int or list of ints")

    # Autonomous change activate?
    autonomous_change_mode = False
    if np.max(allow) > 1000:
        # If there is a value > 1000 in the allow matrix, autonomous change is applied
        autonomous_change_mode = True

    years = range(start_year, end_year+1)
    for year in years:
        print(year)
        i = year - min(years)

        # Check if lus_matrix is xlsx file or not (in this case update lus_matrix by year)
        if lus_matrix_path.endswith(".xlsx"):
            lus_matrix = pd.read_excel(lus_matrix_path).iloc[:, 1:].to_numpy()
        else:
            new_filename = f"{lus_matrix_path}yield_data_{year}.xlsx"
            # Check if the dynamically constructed filename ends with ".xlsx"
            if new_filename.endswith(".xlsx"):
                lus_matrix = pd.read_excel(new_filename).iloc[:, 1:].to_numpy()
            else:
                print(f"No valid file found for year {year}")

        if year in change_years:
            index = change_years.index(year)
            suit_array = rasterio.open(change_paths[index]).read()
            land_array, suit_array = check_no_data_value(land_array, suit_array, no_data_value)

        if i == 0:
            # Initialize demand elasticities array
            dem_elas = np.zeros(len(dem_weights), dtype="float32")
            # Create a timestamped subfolder for each year
            #dem_elas = np.zeros(len(dem_weights), dtype="float32")
            subdir = create_timestamped_subfolder(out_dir)
            log_file_path = os.path.join(subdir, 'logfile.txt')

            log_initial_data(log_file_path=log_file_path, start_year=start_year, end_year=end_year,
                             change_years=change_years, neigh_weights=neigh_weights, conv_res=conv_res,
                             allow=allow, demand=demand, dem_weights=dem_weights, max_iter=max_iter,
                             max_diff_allow=max_diff_allow, totdiff_allow=totdiff_allow, metadata=metadata, lus_conv=lus_conv)

            # Set initial land cover and age arrays
            old_cov = land_array
            if age_array is not None:
                old_age = age_array

        loop = 0
        maxdiff = 1000
        totdiff = 1000

        # Initialize demand elasticities array
        #dem_elas = np.zeros(len(dem_weights), dtype="float32")

        # Generate a random seed for speed calculation
        seed = random.random()
        if seed > 0.9 or seed < 0.001:
            seed = 0.05
        speed = seed

        # Calculate neighbourhood
        neigh_array = calc_neigh(old_cov, width_neigh, neigh_weights)


        # To do:
        # Persistent parallelization:
        # Split arrays before while loop
        # Only report freq from the land cover array to the compare demand function

        # Iterate until convergence or max iterations reached
        while loop < max_iter and (maxdiff > max_diff_allow or totdiff > totdiff_allow):
            # Calculate land cover change
            land_array = calc_change(old_cov, suit_array, region_array, neigh_array, dem_weights, dem_elas,
                                     conv_res, allow, lus_conv, zonal_array, preference_array, preference_weights,age_array)
            # Calculate demand elasticities
            dem_elas, maxdiff, totdiff, diffarr = comp_demand(
                demand[i], land_array, lus_matrix, dem_elas, speed,
                demand_max=demand_max, demand_setback=demand_setback
            )
            loop += 1
            print(f"year: {year}, loop: {loop}, totdiff: {totdiff}, maxdiff: {maxdiff}, differences: {diffarr} ,elasticities: {dem_elas}")
            # Log metadata for each iteration
            log_metadata(log_file_path,
                         f"Year: {year}, loop: {loop}, demand elasticities: {dem_elas}, differences: {diffarr},total difference: {totdiff}, maximum difference: {maxdiff}")
            # log_metadata(log_file_path, f"Land cover shares: {calcRelFreq(land_array, no_data_value)}")
        if loop == max_iter:
            print('Error')
            break
        # Log separator between iterations
        log_metadata(log_file_path, '##########################################')
        new_cov = land_array
        if age_array is not None:
            # Apply bottom-up autonomous changes based on age, if enabled
            if autonomous_change_mode:
                new_cov = autonomous_change(new_cov, old_cov, old_age, allow, no_data_value)
            # Calculate new age array if provided
            new_age = calc_age(old_cov, new_cov, old_age)
            if out_year is None or year in out_year:
                new_age_out = new_age.copy()
                new_age_out[new_age_out == no_data_value] = no_data_out
                outname = os.path.join(subdir, 'age' + str(year) + '.tif')
                writeArray2GeoTIFF(new_age_out, outname, ras_info, no_data_out, crs, dtype)
            old_age = new_age

        # Write new land cover raster
        if out_year is None or year in out_year:
            outname = os.path.join(subdir, 'cov' + str(year) + '.tif')
            # writeArray2Raster(new_cov, outname, ras_info)
            new_cov_out = new_cov.copy()
            new_cov_out[new_cov_out == no_data_value] = no_data_out
            writeArray2GeoTIFF(new_cov_out, outname, ras_info, no_data_out, crs, dtype)
        old_cov = new_cov