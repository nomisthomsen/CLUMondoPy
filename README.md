# CLUMondo-Based Land Use Modeling

This Python-based land use modeling script simulates land cover changes using principles inspired by the original **CLUMondo C++ model**. It accounts for location suitability, neighborhood effects, conversion resistance and allowance rules, and demand-driven land change dynamics.

---

## 📦 Requirements

- Python **3.8+**

### Required Python Packages

- `numpy`
- `pandas`
- `rasterio`
- `gdal`
- `openpyxl`
- `numba`

Install via:

```bash
pip install -r requirements.txt
```

---

## 📥 Input Data & Parameters

You can run the model using either:

- A config file (recommended)
- Command-line arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--land_array` | str | Path to initial land cover raster |
| `--suit_array` | str | Path to suitability stack (one layer per land class) |
| `--region_array` | str | Mask where `1` indicates restricted areas |
| `--neigh_weights` | str | Comma-separated neighborhood weights (e.g., `0.2,0.3,0.5`) |
| `--start_year` | int | First year of simulation |
| `--end_year` | int | Last year of simulation |
| `--demand` | str | Path to demand file (Excel) |
| `--dem_weights` | str | Comma-separated demand weights |
| `--lus_conv` | str | Path to conversion matrix file (Excel) |
| `--lus_matrix_path` | str | Folder or file containing LUS matrices |
| `--conv_res` | str | Comma-separated conversion resistances per class |
| `--allow` | str | Path to allowance matrix (Excel) |
| `--out_dir` | str | Output folder |
| `--crs` | str | EPSG code for output CRS |

... *(Add optional parameters, output files, dynamic suitability description, and troubleshooting as needed)* ...
