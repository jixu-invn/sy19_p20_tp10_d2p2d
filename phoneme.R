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
res.pca <- prcomp(x=train[, -(nb.p+1)], center=TRUE, scale.=TRUE)
plot(cumsum(res.pca$sdev^2 / sum(res.pca$sdev^2)), type="l", ylim=0:1) # plotting cumulative proportion
summary(res.pca)

nb.pca <- 100 # we take n first principal components

train.pca <- as.data.frame(res.pca$x[, 1:nb.pca]) 
train.pca <- cbind(train.pca, train$y)
names(train.pca)[nb.pca+1] <- 'y'
test.pca <- as.data.frame((scale(test[, -(nb.p+1)], center=res.pca$center) %*% res.pca$rotation)[, 1:nb.pca])
test.pca <- cbind(test.pca, test$y)
names(test.pca)[nb.pca+1] <- 'y'

#------Logistic linear regression--------#
library('nnet')
fit <- multinom(y~., data=train, MaxNWts=1500)
#plot(fit$coefficients, type="l")
pred <- predict(fit, newdata = test[, -(nb.p+1)])
score.multinom <- sum(test$y == pred)/nrow(test) # 0.7355126

fit <- multinom(y~., data=train.pca)
pred <- predict(fit, newdata = test.pca)
score.multinom.pca <- sum(test.pca$y == pred)/nrow(test.pca) # 0.9063893

#------SVM------#
# Réglage de C par validation croisée
cv.svm <- function(train, p, kernel) {
  library("kernlab")
  CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
  N<-length(CC)
  M<-10 # nombre de répétitions de la validation croisée
  err<-matrix(0,N,M)
  x <- as.matrix(train[, -(p+1)])
  for(k in 1:M){
    for(i in 1:N){
      if (kernel == "vanilladot") {
        err[i,k]<-cross(ksvm(x=x, y=train$y, 
                             type="C-svc", 
                             kernel=kernel,
                             C=CC[i],
                             cross=5))
      }
      else if (kernel == "rbfdot") {
        err[i,k]<-cross(ksvm(x=x, y=train$y, 
                             type="C-svc", 
                             kernel=kernel,
                             kpar = "automatic",
                             C=CC[i],
                             cross=5))
      }
    }
  }
  Err<-rowMeans(err)
  plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")
  return(CC[which.min(Err)])
}

c.min <- cv.svm(train.pca, nb.pca, kernel="vanilladot") # 0.01

# Calcul de l'erreur de test avec la meilleure valeur de C
fit<-ksvm(x=as.matrix(train.pca[, -(nb.pca+1)]), y=train.pca$y, 
             type="C-svc", 
             kernel="vanilladot",
             C=0.01)
pred<-predict(fit,newdata=test.pca[, -(nb.pca+1)])
score.svm.linear <- sum(pred==test.pca$y)/nrow(test.pca) # 0.9212481

c.min <- cv.svm(train.pca, nb.pca, kernel="rbfdot") # 1
fit<-ksvm(x=as.matrix(train.pca[, -(nb.pca+1)]), y=train.pca$y, 
          type="C-svc", 
          kernel="rbfdot",
          C=c.min)
pred<-predict(fit, newdata=test.pca[, -(nb.pca+1)])
score.svm.gaus <- sum(pred==test.pca$y)/nrow(test.pca) # 0.9182764

library(dplyr)
library(reticulate)
#reticulate::use_python("/usr/local/Cellar/python@2/2.7.15/bin/python2.7", required =TRUE)
reticulate::py_config()
library(tensorflow)
library(keras)
# seed to replicate the same results in others computers
tensorflow::tf$random$set_seed(1729)

#------Artificial Neural Networks with 2 denses (equivalent to natural spline regression)------#
model <- keras_model_sequential()
model %>% layer_dense(units=nb.p, activation="relu", input_shape=nb.p) %>%
  layer_dense(units=nb.class, activation="softmax")
summary(model)
model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
history <- model %>% fit(as.matrix(train[,-(nb.p+1)]), model.matrix(~ -1 + y, data=train),
                        epochs=50, batch_size=50, validation_split=0.3)
model %>% evaluate(as.matrix(test[,-(nb.p+1)]), model.matrix(~ -1 + y, data=test)) # 0.4015191 0.9093611

#pca
model <- keras_model_sequential()
model %>% layer_dense(units=nb.pca, activation="relu", input_shape=nb.pca) %>%
  layer_dense(units=nb.class, activation="softmax")
summary(model)
model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
history <- model %>% fit(as.matrix(train.pca[,-(nb.pca+1)]), model.matrix(~ -1 + y, data=train.pca),
                         epochs=50, batch_size=50, validation_split=0.3)
model %>% evaluate(as.matrix(test.pca[,-(nb.pca+1)]), model.matrix(~ -1 + y, data=test.pca)) # 0.3212432 0.9078752

#------Convolutional Neural Networks------#
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

#pca
n <- 5
m <- as.integer(nb.pca/n)
train.x.3d <- as.matrix(train.pca[,-(nb.pca+1)])
dim(train.x.3d) <- c(nrow(train.pca),m,n)
test.x.3d <- as.matrix(test.pca[,-(nb.pca+1)])
dim(test.x.3d) <- c(nrow(test.pca),m,n)

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
history <- model %>% fit(train.x.3d, model.matrix(~ -1 + y, data=train.pca),
                         epochs=50, batch_size=50, validation_split=0.3)
model %>% evaluate(test.x.3d, model.matrix(~ -1 + y, data=test.pca)) # 0.2372558 0.9153046 


# Save model #
model %>% save_model_hdf5("keras_out.h5")
model.serialize <- serialize_model(load_model_hdf5("keras_out.h5"))
model <- unserialize_model(model.serialize)


