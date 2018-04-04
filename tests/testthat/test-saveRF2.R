set.seed(238943202)
x <- iris[,-1]
y <- iris[, 1]

forest <- forestry(x,
                   y,
                   sample.fraction = .5,
                   ntree = 3,
                   saveable = TRUE)

# save(forest, file = "tests/testthat/forest.Rda")
# load("tests/testthat/forest.Rda", verbose = TRUE)
# str(forest)

# forest@dataframe
# forest@forest
forest <- relinkCPP_prt(forest)
forest@dataframe
forest@forest

s <- forest@forest_R[[1]]$var_id
sum(s[s < 0])

forest@forest_R[[1]]$leaf_idx
