# Basins and triangles

This repository contains the code used for the analysis presented in the paper:  
"Higher-order interactions influence linear stability and basin stability differently",  
by Y. Zhang, P. S. Skardal, F. Battiston, G. Petri, and M. Lucas

<img src="https://github.com/maximelucas/basins_and_triangles/assets/7493360/1728fff7-5d66-4e13-8803-61b5ab19a332" width="45%">


### Contents
- `notebooks/`: notebooks used to produce the figures and simple simulations
- `code/`: modules with functions used in the notebooks and heavier simulation scripts
- `data/`: simulation outputs saved for plotting

### Dependencies

The code was tested for Python 3.9 and the dependencies specified in [requirements.txt](requirements.txt).

In particular, the code relies on the [XGI library](https://github.com/ComplexGroupInteractions/xgi) and the SciPy ODE integration functions.