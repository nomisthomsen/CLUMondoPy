# CLUMondoPy: CLUMondo-Based Land Use Modeling in Python

This Python-based land use modeling script simulates land use and land cover changes (LULCC) using principles inspired by the original **CLUMondo C++ model** created by **Peter Verburg** (link to the [Github repository](https://github.com/VUEG/CLUMondo) and link to the publication by [van Asselen & Verburg 2012](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12331)). 

This repository contains two main sections:

1. The land cover suitability modelling (which is a prerequisite for land use modelling in CLUMondo). You can find the relevant scripts in the folder [Suitability](CLUMondoPy/Suitability/) and the manual as [PDF here](Suitability_Modelling_Manual.pdf).
  
2. The proper LULCC model based on CLUMondo. You can find the relevant scripts in the folder [CLUMondo](CLUMondoPy/CLUMondo) and the manual as [PDF here](CLUMondoPy_Manual.pdf)

---
A figure which gives an overview of the modelling modules can be found in the original publication by [van Asselen & Verburg 2012](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12331).


---

## Requirements

- Python **3.8+**

### Required Python Packages

- `numpy`
- `pandas`
- `geopandas`
- `rasterio`
- `gdal`
- `openpyxl`
- `numba`
- `joblib`
- `scikit-learn`

---

## Authors
Simon Thomsen (simon.thomsen@thuenen.de);
Melvin Lippe (melvin.lippe@thuenen.de)

## License
This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html). Please refer also to the [LICENSE](CLUMondoPy/LICENSE.md) file in this repository.


