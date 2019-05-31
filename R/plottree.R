#' @include forestry.R

# ---plots a tree ----------------------------------------------------------
#' plot
#' @name plot-forestry
#' @title visualize a tree
#' @rdname plot-forestry
#' @description plots a tree in the forest.
#' @param x A forestry x.
#' @param tree.id Specifies the tree number that should be visulaized.
#' @param print.meta_dta Should the data for the plot be printed?
#' @param beta.char.len The length of the beta values in leaf node
#' representation.
#' @param ... additional arguments that are not used.
#' @import glmnet
#' @examples
#' set.seed(292315)
#' rf <- forestry(x = iris[,-1],
#'                y = iris[, 1])
#'
#' plot(x = rf)
#' plot(x = rf, tree.id = 2)
#' plot(x = rf, tree.id = 500)
#'
#' ridge_rf <- forestry(
#'   x = iris[,-1],
#'   y = iris[, 1],
#'   replace = FALSE,
#'   nodesizeStrictSpl = 10,
#'   mtry = 4,
#'   ntree = 1000,
#'   minSplitGain = .004,
#'   ridgeRF = TRUE,
#'   overfitPenalty = 1.65,
#'   linFeats = 1:2)
#'
#' plot(x = ridge_rf)
#' plot(x = ridge_rf, tree.id = 2)
#' plot(x = ridge_rf, tree.id = 1000)
#'
#' @export
#' @import visNetwork
plot.forestry <- function(x, tree.id = 1, print.meta_dta = FALSE,
                          beta.char.len = 30, ...) {
  if (x@ntree < tree.id | 1 > tree.id) {
    stop("tree.id is too large or too small.")
  }

  forestry_tree <- make_savable(x)

  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  split_feat <- forestry_tree@R_forest[[tree.id]]$var_id
  split_val <- forestry_tree@R_forest[[tree.id]]$split_val

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
    num_splitting = ifelse(root_is_leaf, -split_feat[2], 0),
    num_averaging = ifelse(root_is_leaf, -split_feat[1], 0),
    level = 1)
  if (root_is_leaf) {
    split_feat <- split_feat[-(1:2)]
    split_val <- split_val[-1]
  } else {
    split_feat <- split_feat[-1]
    split_val <- split_val[-1]
  }

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
          num_splitting = 0,
          num_averaging = 0,
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
          left_child = NA,
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

  for (i in nrow(node_info):1) {
    if (node_info$num_splitting[i] != 0) {
      node_info$num_splitting[node_info$parent[i]] <-
        node_info$num_splitting[node_info$parent[i]] + node_info$num_splitting[i]

      node_info$num_averaging[node_info$parent[i]] <-
        node_info$num_averaging[node_info$parent[i]] + node_info$num_averaging[i]
    }
  }


  # Save more information about the splits -------------------------------------
  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  node_info$feat_nm <- NA
  node_info$is_cat_feat <- NA
  node_info$cat_split_value <- NA
  if (sum(!is.na(node_info$split_feat)) != 0) {
    node_info$feat_nm <- feat_names[node_info$split_feat]
    cat_feats <- names(forestry_tree@processed_dta$categoricalFeatureCols_cpp)
    node_info$is_cat_feat <- node_info$feat_nm %in% cat_feats
    node_info$cat_split_value <- as.character(node_info$split_val)

    cat_feat_map <- forestry_tree@categoricalFeatureMapping
    if (length(cat_feat_map) > 0) {
      for (i in 1:length(cat_feat_map)) {
        # i = 1
        nodes_with_this_split <-
          node_info$split_feat == cat_feat_map[[i]]$categoricalFeatureCol &
          (!is.na(node_info$split_feat))

        node_info$cat_split_value[nodes_with_this_split] <-
          as.character(cat_feat_map[[i]]$uniqueFeatureValues[
            node_info$split_val[nodes_with_this_split]])
      }
    }
  }


  # get the observation ids per leaf -------------------------------------------
  ave_idx <- forestry_tree@R_forest[[tree.id]]$leafAveidx

  leaf_idx <- list()
  for (leaf_id in node_info$node_id[node_info$is_leaf]) {
    # leaf_id <- 4

    these_idces <- 1:(node_info$num_averaging[node_info$node_id == leaf_id])

    leaf_idx[[leaf_id]] <- ave_idx[these_idces]
    ave_idx <- ave_idx[-these_idces]
  }

  # Prepare data for VisNetwork ------------------------------------------------

  nodes <- data.frame(
    id = node_info$node_id,
    shape = ifelse(node_info$is_leaf,
                   "box", "circle"),
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

  # collect data for leaves ----------------------------------------------------
  nodes$label <- as.character(nodes$label)
  nodes$title <- as.character(nodes$label)


  dta_x <- forestry_tree@processed_dta$processed_x
  dta_y <- forestry_tree@processed_dta$y


  if (forestry_tree@ridgeRF) {
    # ridge forest
    for (leaf_id in node_info$node_id[node_info$is_leaf]) {
      # leaf_id = 5
      ###
      this_ds <- dta_x[leaf_idx[[leaf_id]],
                       forestry_tree@linFeats + 1, drop = FALSE]
      encoder <- onehot::onehot(this_ds)
      remat <- predict(encoder, this_ds)
      ###
      dta_y_leaf <- dta_y[leaf_idx[[leaf_id]]]
      if (length(unique(dta_y_leaf)) == 1) {
        plm_pred <- c(dta_y_leaf[1], rep(0, ncol(remat)))
        r_squared = 1
      } else {
        plm <- glmnet::glmnet(x = remat,
                              y = dta_y_leaf,
                              lambda = forestry_tree@overfitPenalty,
                              alpha	= 0)

        plm_pred <- predict(plm, type = "coef")
        r_squared = plm$dev.ratio
      }

      plm_pred_names <- c("interc", colnames(remat))

      return_char <- character()
      for (i in 1:length(plm_pred)) {
        return_char <- paste0(return_char,
                              substr(plm_pred_names[i], 1, beta.char.len), " ",
                              round(plm_pred[i], 2), "<br>")
      }
      nodes$title[leaf_id] <- paste0(nodes$label[leaf_id],
                                     "<br> R2 = ",
                                     r_squared,
                                     "<br>========<br>",
                                     return_char)
      nodes$label[leaf_id] <- paste0(nodes$label[leaf_id],
                                     "\n R2 = ",
                                     round(r_squared, 3),
                                     "\n=======\nm = ",
                                     round(mean(dta_y[leaf_idx[[leaf_id]]]), 5))
    }
  } else {
    # not ridge forest
    for (leaf_id in node_info$node_id[node_info$is_leaf]) {
      nodes$label[leaf_id] <- paste0(nodes$label[leaf_id], "\n=======\nm = ",
                                     round(mean(dta_y[leaf_idx[[leaf_id]]]), 5))
    }
  }

  # defines a colors -----------------------------------------------------------
  split_vals <- node_info$split_feat
  split_vals <- ifelse(is.na(split_vals), 0, split_vals)
  split_vals <- factor(split_vals)
  color_code <- grDevices::terrain.colors(n = length(feat_names) + 1,
                                          alpha = .7)
  names(color_code) <- as.character(0:(length(feat_names)))
  nodes$color <- color_code[split_vals]

  # Plot the actual node -------------------------------------------------------
  (p1 <-
    visNetwork(
      nodes,
      edges,
      width = "100%",
      height = "800px",
      main = paste("Tree", tree.id)
    ) %>%
    visEdges(arrows = "to") %>%
    visHierarchicalLayout() %>% visExport(type = "pdf", name = "ridge_tree"))

  if (print.meta_dta) print(node_info)
  return(p1)
}
