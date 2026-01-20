# Optifik

Python library to compute a thickness from an interferometric spectrum.

This library replaces [oospectro](https://github.com/sciunto-org/oospectro/).

## Documentation

<https://sciunto-org.github.io/optifik/>

## Publication

We published the methodology in [EPJE Tips and Tricks (open access CC BY)](https://doi.org/10.1140/epje/s10189-025-00545-9).

To cite this paper:
```bibtex
@Article{Ziapkoff2026,
  author       = {V. Ziapkoff, F. Boulogne, A. Salonen, E. Rio},
  date         = {2026},
  journaltitle = {The European Physical Journal E},
  title        = {White light interferometry analysis for measuring thin film thickness down to a few nanometers},
  doi          = {10.1140/epje/s10189-025-00545-9},
  issn         = {1292-895X},
  number       = {1},
  pages        = {4},
  volume       = {49},
  abstract     = {We present a practical white-light interferometric method, supported by an open-source Python library optifik for automated spectrum-to-thickness deduction, enabling foam film measurements down to a few nanometers. We describe three typical spectral scenarii encountered in this method: spectra exhibiting numerous interference fringes, spectra with a moderate number of peaks, and spectra with only a few identifiable features, providing illustrative examples for each case. We also discuss the main limitations of the technique, including spectral range constraints, the necessity of knowing the refractive index, and the influence of spectral resolution and signal quality. Finally, we demonstrate the application of the method in a time-resolved study of a TTAB (tetradecyltrimethylammonium bromide) foam film undergoing elongation and thinning. This method can be adapted to measure any thin non-opaque layer.},
  refid        = {Ziapkoff2026},
}
```


## Installation

The use of pip must be limited to virtualenv


* From PyPI
```
pip install optifik
```

* From tarball
```
pip install /path/to/optifik-0.1.0.tar.gz
```

* From the source code
```
pip install .
```


## For contributors

* Install an editable version
```
pip install -e .
```

* Install dev tools
```
pip install -e ".[dev]"
```

* Run the test suite
```
pytest
```

* Install doc tools
```
pip install -e ".[docs]"
```

* Build the doc
```
sphinx-build -b html docs docs/_build/html
```

## Licence

The source code is released under the GNU General Public License v3.0.
See LICENSE for details.
