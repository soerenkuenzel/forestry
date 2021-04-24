# This package has been moved to https://github.com/forestry-labs/Rforestry
</br></br></br></br></br></br></br></br></br></br></br></br>

[![Travis-CI Build Status](https://travis-ci.org/soerenkuenzel/forestry.svg?branch=master)](https://travis-ci.org/soerenkuenzel/forestry)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2629366.svg)](https://doi.org/10.5281/zenodo.2629366)


## forestry: Provides Functions for Fast Random Forests

Sören Künzel, Edward Liu, Theo Saarinen, Allen Tang, Jasjeet Sekhon

## Introduction

forestry is a fast implementation of Honest Random Forests. 

## How to install
1. The GFortran compiler has to be up to date. GFortran Binaries can be found [here](https://gcc.gnu.org/wiki/GFortranBinaries).
2. The [devtools](https://github.com/hadley/devtools) package has to be installed. You can install it using,  `install.packages("devtools")`.
3. The package contains compiled code, and you must have a development environment to install the development version. You can use `devtools::has_devel()` to check whether you do. If no development environment exists, Windows users download and install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and macOS users download and install [Xcode](https://itunes.apple.com/us/app/xcode/id497799835).
4. The latest development version can then be installed using 
`devtools::install_github("soerenkuenzel/forestry") `.


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

## Monotonic Constraints

A parameter controlling monotonic constraints for features in forestry.

```R
x <- rnorm(150)+5
y <- .15*x + .5*sin(3*x)
data_train <- data.frame(x1 = x, x2 = rnorm(150)+5, y = y + rnorm(150, sd = .4))

monotone_rf <- forestry(x = data_train %>% select(-y),
                        y = data_train$y,
                        monotonicConstraints = c(-1,-1),
                        nodesizeStrictSpl = 5,
                        nthread = 1,
                        ntree = 25)
predict(monotone_rf, feature.new = data_train %>% select(-y))

```


## OOB Predictions

We can return the predictions for the training dataset using only the trees in
which each observation was out of bag. Note that when there are few trees, or a
high proportion of the observations sampled, there may be some observations 
which are not out of bag for any trees. 
The predictions for these are returned NaN.


```R
library(forestry)

# Train a forest
rf <- forestry(x = iris[,-1], 
               y = iris[,1],
               ntree = 500)
               
# Get the OOB predictions for the training set
oob_preds <- getOOBpreds(rf)

# This should be equal to the OOB error
sum((oob_preds -  iris[,1])^2)
getOOB(rf)
```



