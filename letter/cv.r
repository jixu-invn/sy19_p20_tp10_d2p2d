#------Accuracy--------#
accuracy <- function(test.y,pred.y) {
  sum(diag(table(test.y,pred.y)))/length(test.y)
}


#------LDA QDA NB--------#



lda.cv <- function(data, p, subset=NULL) {
  library("MASS")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- lda(y~., data=train, subset=subset)
    pred <- predict(fit, newdata=test[, -(p+1)], subset=subset)
    acc[k] <- accuracy(test$y,pred$class)
  }
  return(acc)
}

qda.cv <- function(data, p, subset=NULL) {
  library("MASS")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- qda(y~., data=train, subset=subset)
    pred <- predict(fit, newdata=test[, -(p+1)], subset=subset)
    acc[k] <- accuracy(test$y,pred$class)   
    message(k)
  }
  return(acc)
}

naivebayes.cv <- function(data, p, subset=NULL) {
  library("naivebayes")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- naive_bayes(y~., data=train, subset=subset)
    pred <- predict(fit, newdata=test[, -(p+1)], subset=subset)
    acc[k] <- accuracy(test$y,pred)   
    message(k)
  }
  return(acc)
}




#------Logistic linear regression--------#
library("nnet")
multinom.cv <- function(data, p) {
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- multinom(y~., data=train)
    pred <- predict(fit, newdata=test[, -(p+1)])
    acc[k] <- accuracy(test$y,pred) 
    message(k)
  }
  return(acc)
}


#------GAM------#
library("splines")
library('nnet')
gam.cv <- function(data, p, nclass) {
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  
  fm <- paste('ns(', names(data)[1:p], ')', sep = "", collapse = ' + ')
  fm <- as.formula(paste('y~ ', fm))
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- multinom(formula=fm, data=train)
    pred <- predict(fit, newdata=test[, -(p+1)])
    acc[k] <- accuracy(test$y,pred) 
    message(k)
  }
  return(acc)
}



#------Tree & Bagging------#

rpart.cv <- function(data, p) {
  library("rpart")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- rpart(y~., data=train, 
                 method = "class", 
                 control = rpart.control(xval=10, minbucket=10, cp=0.00))
    pred <- predict(fit, newdata=test[, -(p+1)], type="class")
    acc[k] <- accuracy(test$y,pred) 
    message(k)
  }
  return(acc)
}

prune.cv <- function(data, p) {
  library("rpart")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- rpart(y~., data=train, 
                 method = "class", 
                 control = rpart.control(xval=10, minbucket=10, cp=0.00))
    i.min <- which.min(fit$cptable[, 4])
    cp.opt <- fit$cptable[i.min, 1]
    fit.prune <- prune(fit, cp=cp.opt)
    pred <- predict(fit.prune, newdata=test[, -(p+1)], type="class")
    acc[k] <- accuracy(test$y,pred) 
    message(k)
  }
  return(acc)
}

bagged.cv <- function(data, p) {
  library("randomForest")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- randomForest(y~., data=train, mtry=p)
    pred <- predict(fit, newdata=test[, -(p+1)], type="response")
    acc[k] <- accuracy(test$y,pred) 
    message(k)
  }
  return(acc)
}

rf.ntree.cv <- function(data, p) {
  library("randomForest")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  
  train <- data[folds!=1,]
  test <- data[folds==1,]
  trees <- c(2000,3000,4000)
  acc <- rep(0, length(trees))
  for (t in 1:length(trees)) {
    fit <- randomForest(y~.,ntree=trees[t], data=train)
    pred <- predict(fit, newdata=test[, -(p+1)], type="response")
    acc[t] <- accuracy(test$y,pred) 
    message(trees[t])
  }
  plot(x = trees,y = acc)
  return(acc)
}

rf.cv <- function(data, p, trees) {
  library("randomForest")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- randomForest(y~.,ntree=trees, data=train)
    pred <- predict(fit, newdata=test[, -(p+1)], type="response")
    acc[k] <- accuracy(test$y,pred) 
    message(k)
    message(acc[k])
  }
  return(acc)
}


#------SVM------#
svm.c.cv <- function(data, p, kernel) {
  library("kernlab")
  #CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
  #CC<-c(80,100,110)
  CC<-c(0.01,0.1,1,10,100)
  N<-length(CC)
  M<-10 # nombre de répétitions de la validation croisée
  err<-matrix(0,N)
  x <- as.matrix(data[, -(p+1)])
  for(i in 1:N){
      if (kernel == "vanilladot") {
        err[i]<-cross(ksvm(x=x, y=data$y, 
                             type="C-svc", 
                             kernel=kernel,
                             C=CC[i],
                             cross=5))
      }
      else if (kernel == "rbfdot") {
        err[i]<-cross(ksvm(x=x, y=data$y, 
                             type="C-svc", 
                             kernel=kernel,
                             kpar = "automatic",
                             C=CC[i],
                             cross=5))
      }
      else if (kernel == "polydot") {
        res <- ksvm(x=x, y=data$y, 
                    type="C-svc", 
                    kernel=kernel,
                    C=CC[i],
                    cross=5)
        print(res)
        err[i]<-cross(res)
      }
    
      message(CC[i])
      message(err[i])
    }
    
  
  #Err<-rowMeans(err)
  plot(CC,err,type="b",log="x",xlab="C",ylab="CV error")
  return(CC[which.min(err)])
}

svm.cv <- function(data, p, kernel, c) {
  library("kernlab")
  K <- 10
  n <- nrow(data)
  set.seed(1212)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- ksvm(x=as.matrix(train[, -(p+1)]), y=train$y, 
                type="C-svc", 
                kernel=kernel,
                #kpar = "automatic",
                C=c)
    pred <- predict(fit, newdata=test[, -(p+1)])
    acc[k] <- accuracy(test$y,pred) 
    message(k)
  }
  return(acc)
}

#------Neural Networks------#
nn.cv <- function(data, n.dim, n.class) {
  library(dplyr)
  library(reticulate)
  reticulate::use_python("C:/Users/tbour/AppData/Local/r-miniconda/envs/r-reticulate/python.exe", required =TRUE)
  reticulate::py_config()
  library(tensorflow)
  library(keras)
  tensorflow::tf$random$set_seed(1212)
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  acc <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    model <- keras_model_sequential()
    model %>% 
      layer_dense(units = 128, activation = 'relu', input_shape = n.dim-1) %>% 
      #layer_dropout(0.4) %>%
      layer_dense(units = 64, activation = 'relu') %>%
      #layer_dropout(0.2) %>%
      layer_dense(units = n.class, activation = 'softmax')
    
    model %>% compile(
      loss="categorical_crossentropy", 
      optimizer='adam', 
      metrics='accuracy')
    
    model %>% fit(
      as.matrix(train[,-n.dim]), 
      model.matrix(~ -1 + y, data=train),
      epochs=30, 
      batch_size=128)
    #,validation_split=0.2
    pred <- model %>% evaluate(as.matrix(test[,-n.dim]), model.matrix(~ -1 + y, data=test))
    acc[k] <- pred[2]
  }
  return(acc)
}

