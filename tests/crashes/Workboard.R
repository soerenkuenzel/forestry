# Workboard

# Set seed for reproductivity
set.seed(292313)

# Use Iris Data
# Add a location column to iris dataset
iris$location <- as.factor(sample(c("north", "south", "east", "west"),
                                  nrow(iris),
                                  replace = TRUE))
test_idx <- sample(nrow(iris), 10)
x_train <- iris[-test_idx, -1]
y_train <- iris[-test_idx, 1]
x_test <- iris[test_idx, -1]

# Create a random forest
rf <- forestry(x = x_train, y = y_train, nthread = 1)

pred <- predict(rf, x_test)
print(pred)

# Testing C++ implementation:
distances <- compute_rf_dist(rf, x_test, p = 1, distance.feat = "Petal.Length")
print(distances)



