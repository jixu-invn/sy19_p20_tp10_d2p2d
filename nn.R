library(dplyr)
library(reticulate)
reticulate::use_python("/Users/jl-x/Library/r-miniconda/envs/r-reticulate/bin/python3.8", required =TRUE)
reticulate::py_config()
library(tensorflow)
library(keras)
# seed to replicate the same results in others computers
tensorflow::tf$random$set_seed(1729)

#------Artificial Neural Networks------#
ann.cv <- function(data, n.in, n.out) {
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    model <- keras_model_sequential()
    model %>% layer_dense(units=n.in, activation="relu", input_shape=n.in) %>%
      layer_dense(units=n.out, activation="softmax")
    model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
    model %>% fit(as.matrix(train[,-(n.in+1)]), model.matrix(~ -1 + y, data=train),
                             epochs=30, batch_size=32, validation_split=0.3)
    res <- model %>% evaluate(as.matrix(test[,-(n.in+1)]), model.matrix(~ -1 + y, data=test))
    err[k] <- res[2]
  }
  return(err)
}

#------Deep Neural Networks------#
dnn.cv <- function(data, n.in, n.out) {
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    model <- keras_model_sequential()
    model %>% layer_dense(units=n.in, activation="relu", input_shape=n.in) %>%
      layer_dense(units=n.in*2, activation="relu") %>%
      layer_dropout(0.5) %>%
      layer_dense(units=as.integer(n.in/2), activation="relu") %>%
      layer_dropout(0.5) %>%
      layer_dense(units=n.out, activation="softmax")
    model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
    model %>% fit(as.matrix(train[,-(n.in+1)]), model.matrix(~ -1 + y, data=train),
                  epochs=30, batch_size=32, validation_split=0.3)
    res <- model %>% evaluate(as.matrix(test[,-(n.in+1)]), model.matrix(~ -1 + y, data=test))
    err[k] <- res[2]
  }
  return(err)
}

#------Convolutional Neural Networks------#
cnn.cv <- function(data, n.in, n.out, n.dim=16) {
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    n.y <- as.integer(n.in/n.dim)
    train.x <- as.matrix(train[, -(n.in+1)])
    dim(train.x) <- c(nrow(train), n.y, n.dim)
    test.x <- as.matrix(test[, -(n.in+1)])
    dim(test.x) <- c(nrow(test), n.y, n.dim)
    
    model <- keras_model_sequential()
    model %>% layer_conv_1d(filters=16, kernel_size=3, activation="relu", input_shape=c(n.y,n.dim)) %>%
      layer_average_pooling_1d(pool_size=2) %>%
      layer_conv_1d(filters=32, kernel_size=3, activation="relu") %>%
      layer_average_pooling_1d(pool_size=2) %>%
      layer_dropout(0.5) %>%
      layer_flatten() %>%
      layer_dense(units=128, activation="relu") %>%
      layer_dropout(0.5) %>%
      layer_dense(units=nb.class, activation="softmax")
    model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
    model %>% fit(train.x, model.matrix(~ -1 + y, data=train),
                  epochs=30, batch_size=32, validation_split=0.3)
    res <- model %>% evaluate(test.x, model.matrix(~ -1 + y, data=test))
    err[k] <- res[2]
  }
  return(err)
}

