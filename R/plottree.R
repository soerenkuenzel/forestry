#' @include forestry.R

# ---visulizes a tree ----------------------------------------------------------
#' plot
#' @name plot-forestry
#' @title visualize a tree
#' @rdname plot-forestry
#' @description visulizes a tree in the forest.
#' @param x A forestry x.
#' @param tree.id Specifies the tree number that should be visulaized.
#' @param print.meta_dta Should the data for the plot be printed?
#' @param ... additional arguments that are not used.
#' @examples
#' set.seed(292315)
#' library(forestry)
#' test_idx <- sample(nrow(iris), 3)
#' x_train <- iris[-test_idx, -1]
#' y_train <- iris[-test_idx, 1]
#' x_test <- iris[test_idx, -1]
#'
#' rf <- forestry(x = x_train, y = y_train)
#'
#' plot(rf)
#' @export
#' @import visNetwork
plot.forestry <- function(x, tree.id = 1, print.meta_dta = FALSE, ...) {
  if (class(x)[[1]] != "forestry") {
    stop("x must be a forestry x")
  }
  if (x@ntree < tree.id | 1 > tree.id) {
    stop("tree.id is too large or too small.")
  }

  forestry_tree <- make_savable(x)

  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  split_feat <- forestry_tree@R_forest[[tree.id]]$var_id
  split_val <- forestry_tree@R_forest[[tree.id]]$split_val

  split_feat1 <- split_feat
  split_val1 <- split_val

  # get info for the first node ------------------------------------------------
  root_is_leaf <- split_feat[1] < 0
  node_info <- data.frame(
    node_id = 1,
    is_leaf = root_is_leaf,
    parent = NA,
    left_child = ifelse(root_is_leaf, NA, 2),
    right_child = NA,
    split_feat = ifelse(root_is_leaf, NA, split_feat[1]),
    split_val = ifelse(root_is_leaf, NA, split_val[1]),
    num_splitting = NA,
    num_averaging = NA,
    level = 1)
  split_feat <- split_feat[-1]
  split_val <- split_val[-1]

  split_feat1
  # loop through split feat to get all the information -------------------------
  while (length(split_feat) != 0) {
    if (!node_info$is_leaf[nrow(node_info)]) {
      # previous node is not leaf => left child of previous node
      parent <- nrow(node_info)
      node_info$left_child[parent] <- nrow(node_info) + 1
    } else {
      # previous node is leaf => right child of last unfilled right
      parent <-
        max(node_info$node_id[!node_info$is_leaf &
                                is.na(node_info$right_child)])
      node_info$right_child[parent] <- nrow(node_info) + 1
    }
    if (split_feat[1] > 0) {
      # it is not a leaf
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = FALSE,
          parent = parent,
          left_child = nrow(node_info) + 2,
          right_child = NA,
          split_feat = split_feat[1],
          split_val = split_val[1],
          num_splitting = NA,
          num_averaging = NA,
          level = node_info$level[parent] + 1
        )
      )
      split_feat <- split_feat[-1]
      split_val <- split_val[-1]
    } else {
      #split_feat[1] < 0
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = TRUE,
          parent = parent,
          left_child = nrow(node_info) + 2,
          right_child = NA,
          split_feat = NA,
          split_val = NA,
          num_splitting = -split_feat[2],
          num_averaging = -split_feat[1],
          level = node_info$level[parent] + 1
        )
      )
      split_feat <- split_feat[-(1:2)]
      split_val <- split_val[-1]
    }
  }

  # Save more information about the splits -------------------------------------
  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  node_info$feat_nm <- feat_names[node_info$split_feat]
  cat_feats <- names(forestry_tree@processed_dta$categoricalFeatureCols_cpp)
  node_info$is_cat_feat <- node_info$feat_nm %in% cat_feats
  node_info$cat_split_value <- as.character(node_info$split_val)

  cat_feat_map <- forestry_tree@categoricalFeatureMapping
  if (length(cat_feat_map) > 0) {
    for (i in 1:length(cat_feat_map)) {
      # i <- 1
      nodes_with_this_split <-
        node_info$split_feat == cat_feat_map[[i]]$categoricalFeatureCol &
        (!is.na(node_info$split_feat))

      node_info$cat_split_value[nodes_with_this_split] <-
        as.character(cat_feat_map[[i]]$uniqueFeatureValues[
          node_info$split_val[nodes_with_this_split]])
    }
  }
  node_info
  # Prepare data for VisNetwork ------------------------------------------------

  nodes <- data.frame(
    id = node_info$node_id,
    shape = ifelse(node_info$is_leaf,
                   "square", "circle"),
    label = ifelse(
      node_info$is_leaf,
      paste0(node_info$num_averaging, " Obs"),
      paste0(feat_names[node_info$split_feat])
    ),
    level = node_info$level
  )

  edges <- data.frame(
    from = node_info$parent,
    to = node_info$node_id,
    smooth = list(
      enabled = TRUE,
      type = "cubicBezier",
      roundness = .5
    )
  )
  edges <- edges[-1, ]


  edges$label =
    ifelse(
      floor(node_info$split_val[edges$from]) == node_info$split_val[edges$from],
      ifelse(
        node_info$left_child[edges$from] == edges$to,
        paste0(" = ", node_info$cat_split_value[edges$from]),
        paste0(" != ", node_info$cat_split_value[edges$from])
      ),
      ifelse(
        node_info$left_child[edges$from] == edges$to,
        paste0(" < ", round(node_info$split_val[edges$from], digits = 2)),
        paste0(" >= ", round(node_info$split_val[edges$from], digits = 2))
      )
    )

  edges$width = node_info$num_averaging[edges$to] /
    (node_info$num_averaging[1] / 4)

  p1 <-
    visNetwork(
      nodes,
      edges,
      width = "100%",
      height = "1300px",
      main = paste("Tree", tree.id)
    ) %>%
    visEdges(arrows = "to") %>%
    visHierarchicalLayout() %>% visExport(type = "pdf", name = "ridge_tree")

  if (print.meta_dta) print(node_info)
  return(p1)
}
