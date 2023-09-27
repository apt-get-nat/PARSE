# PARSE
The Plasma-prescribed Active Region Static Extrapolation (PARSE) dataset consists of approximagely 7,000 magnetohydrostatic simulations of coronal regions, based on almost 1,000 [SHARP images](https://github.com/mbobra/SHARPs). The intent is to provide these solutions to the community, for testing, validation and machine learning training. The extrapolations were done with a [RBF-FD-based magnetohydrostatic solver](https://github.com/apt-get-nat/RBF-MHS).

Please find the paper describing this dataset [here](https://arxiv.org/abs/2308.02138). The dataset itself is available on [zenodo](https://zenodo.org/record/8213061).

A small python packages is included here, which can be installed with
```
pip install git+https://github.com/apt-get-nat/PARSE
```
which can import, interpolate and trace individual PARSE realizations.
