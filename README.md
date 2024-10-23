
# Delight

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/delight?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/delight/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/delight/smoke-test.yml)](https://github.com/LSSTDESC/delight/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/LSSTDESC/delight/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/delight)
[![Read The Docs](https://img.shields.io/readthedocs/delight)](https://delight.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/LSSTDESC/delight/asv-main.yml?label=benchmarks)](https://LSSTDESC.github.io/delight/)




**Photometric redshift via Gaussian processes with physical kernels.**

Read the documentation here: [http://delight.readthedocs.io](http://delight.readthedocs.io)

*Warning: this code is still in active development and is not quite ready to be blindly applied to arbitrary photometric galaxy surveys. But this day will come.*

[![alt tag](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/ixkael/Delight/blob/master/LICENSE)
[![alt tag](https://travis-ci.org/ixkael/Delight.svg?branch=master)](https://travis-ci.org/ixkael/Delight)
[![Documentation Status](https://readthedocs.org/projects/delight/badge/?version=latest&style=flat)](http://delight.readthedocs.io/en/latest/?badge=latest)
[![Latest PDF](https://img.shields.io/badge/PDF-latest-orange.svg)](https://github.com/ixkael/Delight/blob/master/paper/PhotoZviaGP_paper.pdf)
[![Coverage Status](https://coveralls.io/repos/github/ixkael/Delight/badge.svg?branch=master)](https://coveralls.io/github/ixkael/Delight?branch=master)

**Tests**: pytest for unit tests, PEP8 for code style, coveralls for test coverage.

## Content

**./paper/**: journal paper describing the method </br>
**./delight/**: main code (Python/Cython) </br>
**./tests/**: test suite for the main code </br>
**./notebooks/**: demo notebooks using delight </br>
**./data/**: some useful inputs for tests/demos </br>
**./docs/**: documentation </br>
**./other/**: useful mathematica notebooks, etc </br>

## Requirements

Python 3.5, cython, numpy, scipy, pytest, pylint, coveralls, matplotlib, astropy, mpi4py </br>

## Authors

Boris Leistedt (NYU) </br>
David W. Hogg (NYU) (Flatiron)

Please cite [Leistedt and Hogg (2016)]
(https://arxiv.org/abs/1612.00847) if you use this code your
research. The BibTeX entry is:

    @article{delight,
        author  = "Boris Leistedt and David W. Hogg",
        title   = "Data-driven, Interpretable Photometric Redshifts Trained on Heterogeneous and Unrepresentative Data",
        journal = "The Astrophysical Journal",
        volume  = "838",
        number  = "1",
        pages   = "5",
        url     = "http://stacks.iop.org/0004-637X/838/i=1/a=5",
        year    = "2017",
        eprint         = "1612.00847",
        archivePrefix  = "arXiv",
        primaryClass   = "astro-ph.CO",
        SLACcitation   = "%%CITATION = ARXIV:1612.00847;%%"
    }


## License

Copyright 2016-2017 the authors. The code in this repository is released under the open-source MIT License. See the file LICENSE for more details.


## Installation and maintenance of this package

This package is maintained by the LSSTDESC collaboration and the DESC-RAIL team.
This project in handle under the LINCC-Framework.


### Usual installation

The package can be installed with a single command `pip`:




```
>> pip install .
```


or


```
>> pip install -e .
```

### Run the tests

#### Basic tests

Very basic tests can be run from top level of `Delight` package using the scripts in `scripts/` as follow:

```
python scripts/processFilters.py tests/parametersTest.cfg
python scripts/processSEDs.py tests/parametersTest.cfg
python scripts/simulateWithSEDs.py tests/parametersTest.cfg
```

#### Unitary tests

```
pytest -v tests/*.py
```

### Install the Delight documentation

- install the python package ``pandoc``either with conda or with pip,
- install sphinx packages as follow:

Either do at top level (same as ``pyproject.toml``) by selecting the packages under the ``[doc]`` section inside
the ``pyproject.toml`` project configuration  file

```
>> pip install -e .'[doc]'
```

or under ``docs/``  by selecting the sphinx packages specified in the ``requirements.txt`` file :

```
>> pip install -r requirements.txt
```

Then run the sphinx command accordig the [sphinx documentation](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html):

```
>> python -m sphinx -T -E -b html -d _build/doctrees -D language=en . ../_readthedocs/html
```

And open the sphinx documentation:

```
>> open ../_readthedocs/html/index.html 
```

## More on LINCC Framework


This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> ./.setup_dev.sh
>> conda install pandoc
```

Notes:
1. `./.setup_dev.sh` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
2. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
