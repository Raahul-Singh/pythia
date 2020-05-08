Pythia
======

Pythia is an open-source Python library for Solar Magnetic Active Region Data Analysis.
This library holds the principal work done as part of the [OpenAstonomy GSoC'20 project](https://summerofcode.withgoogle.com/projects/#5503197600284672), **Solar Weather Forecasting using Linear Algebra**

Description
-----------
The goal of this project is to develop a model for forecasting the likelihood that an Active Region on the sun would produce a solar flare in the near future. The dataset used in this project is the [Sunspotter dataset](https://zenodo.org/record/1478972#.XrUPH_HhU5l), which includes a complexity score for each AR. It is to be explored if the complexity of the AR corresponds to a higher probability of flare production. In the course of this project, a Search Events object capable of querying HEK and HELIO databases will be created.

Installation
------------

Use git to grab the latest version of Pythia:

    git clone https://github.com/Raahul-Singh/pythia.git

Done! In order to enable Pythia to be imported from any location you must make
sure that the library is somewhere in your PYTHONPATH environmental variable.
For now the easiest thing is to install it locally by running,
```
pip install -e .
```
from the directory you just
downloaded. 

