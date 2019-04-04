[![Travis-CI Build Status](https://travis-ci.org/soerenkuenzel/forestry.svg?branch=master)](https://travis-ci.org/soerenkuenzel/forestry)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2629366.svg)](https://doi.org/10.5281/zenodo.2629366)


## forestry: Provides Functions for Fast Random Forests

Sören Künzel, Edward Liu, Theo Saarinen, Allen Tang, Jasjeet Sekhon

## Introduction

forestry is a fast implementation of Honest Random Forests. 

## How to install

The latest development version can be installed directly from Github using [devtools](https://github.com/hadley/devtools):

```R
if (!require("devtools")) install.packages("devtools")
devtools::install_github("soerenkuenzel/forestry")
```

The package contains compiled code, and you must have a development environment to install the development version. (Use `devtools::has_devel()` to check whether you do.) If no development environment exists, Windows users download and install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and macOS users download and install [Xcode](https://itunes.apple.com/us/app/xcode/id497799835).

## Usage 

```R
set.seed(292315)
library(forestry)
test_idx <- sample(nrow(iris), 3)
x_train <- iris[-test_idx, -1]
y_train <- iris[-test_idx, 1]
x_test <- iris[test_idx, -1]

rf <- forestry(x = x_train, y = y_train)
weights = predict(rf, x_test, aggregation = "weightMatrix")$weightMatrix

weights %*% y_train
predict(rf, x_test)
```

## Ridge Random Forest

A fast implementation of random forests using ridge penalized splitting and ridge regression for predictions.

Example:

```R
set.seed(49)
library(forestry)

n <- c(100)
a <- rnorm(n)
b <- rnorm(n)
c <- rnorm(n)
y <- 4*a + 5.5*b - .78*c
x <- data.frame(a,b,c)
forest <- forestry(x, y, ridgeRF = TRUE)
predict(forest, x)
```
