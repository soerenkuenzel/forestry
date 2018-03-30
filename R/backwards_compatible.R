# We are making sure here that RF is backwards compatible


# -- honestRF ------------------------------------------------------------------
#' @title Honest Random Forest
#' @description This function is deprecated and only exists for backwards
#' backwards compatibility. The function you want to use is `forestry`.
#' @inheritParams forestry
#' @export honestRF
honestRF <- function(...) forestry(...)


#' @title Honest Random Forest
#' @description This function is deprecated and only exists for backwards
#' backwards compatibility. The function you want to use is `autoforestry`.
#' @inheritParams autoforestry
#' @export autohonestRF
autohonestRF <- function(...) autoforestry(...)
