# We are making sure here that RF is backwards compatible


# -- honestRF ------------------------------------------------------------------
#' @title Honest Random Forest
#' @description This function is deprecated and only exists for backwards
#' backwards compatibility. The function you want to use is `forestry`.
#' @inheritParams forestry
#' @param ... parameters which are passed directly to `forestry`
#' @export honestRF
honestRF <- function(...) forestry(...)


#' @title Honest Random Forest
#' @description This function is deprecated and only exists for backwards
#' backwards compatibility. The function you want to use is `autoforestry`.
#' @inheritParams autoforestry
#' @param ... parameters which are passed directly to `autoforestry`
#' @export autohonestRF
autohonestRF <- function(...) autoforestry(...)


# -- Random Forest Constructor -------------------------------------------------
#' @title forestry class
#' @name forestry-class
#' @description `honestRF` class only exists for backwards compatibility reasons
setClass(Class = "honestRF",
         contains = 'forestry')
