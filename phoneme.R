#------Read Data--------#
data <- read.table("data/phoneme_train.txt", header = TRUE)
data$y <- as.factor(data$y)
summary(data$y)
nb.p <- ncol(data)-1
nb.class <- nlevels(data$y)

#------Split Train/Test Set--------#
library(caret)
set.seed(1729)
train.index <- createDataPartition(y=data$y, p=0.7, list=FALSE)
train <- data[train.index,]
test <- data[-train.index,]

#------Principle component analysis--------#
res.pca <- prcomp(x=data.train[names(data.train)!='y'], center=TRUE, scale.=TRUE)
plot(cumsum(res.pca$sdev^2 / sum(res.pca$sdev^2)), type="l", ylim=0:1) # plotting cumulative proportion
summary(res.pca)

library('nnet')
fit <- multinom(y~., data=data.train, MaxNWts=1500)
#plot(fit$coefficients, type="l")
pred <- predict(fit, newdata = data.test[names(data.test)!="y"])
score.multinom <- sum(data.test$y == pred)/nrow(data.test)

library(dplyr)
library(reticulate)
#reticulate::use_python("/usr/local/Cellar/python@2/2.7.15/bin/python2.7", required =TRUE)
reticulate::py_config()
library(tensorflow)
library(keras)
# seed to replicate the same results in others computers
tensorflow::tf$random$set_seed(1729)

# Artificial Neural Networks with 2 denses (equivalent to natural spline regression) #
model <- keras_model_sequential()
model %>% layer_dense(units=nb.p, activation="relu", input_shape=nb.p) %>%
  layer_dense(units=nb.class, activation="softmax")
summary(model)
model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
history <- model %>% fit(as.matrix(train[,-(nb.p+1)]), model.matrix(~ -1 + y, data=train),
                        epochs=50, batch_size=50, validation_split=0.3)
model %>% evaluate(as.matrix(test[,-(nb.p+1)]), model.matrix(~ -1 + y, data=test)) # 0.4015191 0.9093611

# Convolutional Neural Networks #
n <- 16
m <- as.integer(nb.p/n)
train.x.3d <- as.matrix(train[,-(nb.p+1)])
dim(train.x.3d) <- c(nrow(train),m,n)
test.x.3d <- as.matrix(test[,-(nb.p+1)])
dim(test.x.3d) <- c(nrow(test),m,n)

model <- keras_model_sequential()
model %>% layer_conv_1d(filters=16, kernel_size=3, activation="relu", input_shape=c(m,n)) %>%
  layer_max_pooling_1d(pool_size=2) %>%
  layer_conv_1d(filters=32, kernel_size=3, activation="relu") %>%
  layer_max_pooling_1d(pool_size=2) %>%
  layer_dropout(0.2) %>%
  layer_flatten() %>%
  layer_dense(units=128, activation="relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units=nb.class, activation="softmax")
model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
history <- model %>% fit(train.x.3d, model.matrix(~ -1 + y, data=train),
                         epochs=50, batch_size=50, validation_split=0.3)
model %>% evaluate(test.x.3d, model.matrix(~ -1 + y, data=test)) # 0.3056337 0.9108469 

# Save model #
model %>% save_model_hdf5("keras_out.h5")
model.serialize <- serialize_model(load_model_hdf5("keras_out.h5"))
model <- unserialize_model(model.serialize)


