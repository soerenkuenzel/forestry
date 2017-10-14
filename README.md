[![Travis-CI Build Status](https://travis-ci.org/soerenkuenzel/forestry.svg?branch=master)](https://travis-ci.org/soerenkuenzel/forestry)


`forestry` provides functions for estimating heterogenous treatment effects. 

## How to install

The latest development version can be installed directly from Github using [devtools](https://github.com/hadley/devtools):

```R
if (!require("devtools")) install.packages("devtools")
devtools::install_github("soerenkuenzel/forestry")
```

The package contains compiled code, and you must have a development environment to install the development version. (Use `devtools::has_devel()` to check whether you do.) If no development environment exists, Windows users download and install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and macOS users download and install [Xcode](https://itunes.apple.com/us/app/xcode/id497799835).

## TODO
1. replace honestRF by randomforest
1. rename the files which only run the R code such that they can be easily 
identified.


