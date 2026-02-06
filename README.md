# CLUMondoPy: CLUMondo-Based Land Use Modeling in Python

This Python-based land use modeling script simulates land use and land cover changes (LULCC) using principles inspired by the original **CLUMondo C++ model** created by **Peter Verburg** (link to the [Github repository](https://github.com/VUEG/CLUMondo) and link to the publication by [van Asselen & Verburg 2012](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12331)). 

This repository contains two main sections:

1. The land cover suitability modelling (which is a prerequisite for land use modelling in CLUMondo). You can find the relevant scripts in the folder [Suitability](CLUMondoPy/Suitability/) and the manual as [PDF here](Suitability_Modelling_Manual.pdf).
  
2. The proper LULCC model based on CLUMondo. You can find the relevant scripts in the folder [CLUMondo](CLUMondoPy/CLUMondo) and the manual as [PDF here](CLUMondoPy_Manual.pdf)

A figure which gives an overview of the modelling modules can be found in the original publication by [van Asselen & Verburg 2012](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12331).


---

## Repository Structure (important)

The Python source code lives inside the `CLUMondoPy/` folder.

✅ **To run the model without installing the package**, change directory into `CLUMondoPy/` first (see Quickstart below).  
✅ Alternatively, install the repo in editable mode (`pip install -e .`) and run from anywhere.


## Requirements

- Python **3.8+** (recommended: **3.10/3.11**)
- Recommended environment manager: **Miniforge / Anaconda / Mambaforge** (Conda-based)

### Required Python packages

Core dependencies include:
- `numpy`, `pandas`, `geopandas`
- `rasterio`, `gdal`
- `openpyxl`
- `numba`, `joblib`
- `scikit-learn`, `statsmodels`, `scipy`, `xgboost`

> Note: Geospatial packages (`gdal`, `rasterio`, `geopandas`) are often easiest to install via **conda-forge**.

---
## Installation

### Recommended option: Conda / Miniforge (geospatial-friendly)

Create an environment (example name: `clupy`) and install dependencies.

```bash
conda create -n clupy python=3.10
conda activate clupy
```

Install dependencies (recommended via conda-forge for geospatial packages):
```bash
conda install -c conda-forge numpy pandas geopandas rasterio gdal openpyxl numba joblib scikit-learn
```

Optional (recommended for development / stable imports):
```bash
pip install -e .
```
---
## Quickstart: Run CLUMondoPy
The main entrypoint is:
- `CLUMondoPy/Scripts/run_CLUMondoPy.py`
The script calls model logic from:
- `CLUMondoPy/CLUMondo/`

### 1) Run without installing (simple)
**Always run from the `CLUMondoPy/` directory** (the folder containing `CLUMondo/`, `Scripts/`, `Suitability/`)

**Windows (Powershell/ Windows Terminal):**
```powershell
cd "\path\to\repo-root\CLUMondoPy"
conda run --no-capture-output -n clupy python -m Scripts.run_CLUMondoPy --config "path\to\config_file.txt"
```

**Windows (Anaconda Prompt/ Miniforge Prompt):**
```bat
cd \path\to\repo-root\CLUMondoPy
conda activate clupy
python -m Scripts.run_CLUMondoPy --config "path\to\config_file.txt"
```

**Linux/ HPC (bash):**
```bash
cd /path/to/repo-root/CLUMondoPy
conda activate clupy
python -m Scripts.run_CLUMondoPy --config "/path/to/config_file.txt"
```

### 2) Run after installing (`pip install -e .`)
If you installed in editable mode, you can still run using -m (recommended), and you are less sensitive to the current directory:
```bash
python -m Scripts.run_CLUMondoPy --config "/path/to/config_file.txt"
```
---
## Configuration

The model is executed using a --config text file. The config typically contains:

- file paths to required inputs (rasters, Excel tables, etc.)
- model/scenario parameters
- output paths or output directory settings

Please refer to the [manual](CLUMondoPy_Manual.pdf) for a complete list of input requirements for the configuration txt file.

### Path notes

**Windows:** quote paths, especially if they contain spaces.

Both `E:\folder\file.txt` and `E:/folder/file.txt` are typically fine for Python.

---

## Suitability Modelling
Suitability modelling is a prerequisite for CLUMondo-based land use modelling with CLUMondoPy.
- Code: `CLUMondoPy/Suitability/`
- [Manual](Suitability_Modelling_Manual.pdf) 
---

## Authors
Simon Thomsen (simon.thomsen@thuenen.de);
Melvin Lippe (melvin.lippe@thuenen.de)

## License
This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html). Please refer also to the [LICENSE](CLUMondoPy/LICENSE.md) file in this repository.


