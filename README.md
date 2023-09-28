# Deeper but smaller: Higher-order interactions increase linear stability but shrink basins

This repository contains the code used for the analysis presented in the paper:  
"Deeper but smaller: Higher-order interactions increase linear stability but shrink basins",  
by Y. Zhang, P. S. Skardal, F. Battiston, G. Petri, and M. Lucas

<img src="https://github.com/maximelucas/basins_and_triangles/files/12754394/figure_2a.pdf" width="45%">


### Contents
- `notebooks/`: notebooks used to produce the figures and simple simulations
- `code/`: modules with functions used in the notebooks and heavier simulation scripts
- `data/`: simulation outputs saved for plotting

### Dependencies

The code was tested for Python 3.9 and the dependencies specified in [requirements.txt](requirements.txt).

In particular, the code relies on the [XGI library](https://github.com/ComplexGroupInteractions/xgi) and the [SciPy](https://github.com/scipy/scipy) ODE integration functions.