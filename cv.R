# err rate of lda using cross validation (non-nested)
lda.cv <- function(data, p, subset=NULL) {
  library("MASS")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- lda(y~., data=train, subset=subset)
    pred <- predict(fit, newdata=test[, -(p+1)], subset=subset)
    err[k] <- mean(pred$class == test$y)   
  }
  return(err)
}

# err rate of qda using cross validation (non-nested)
qda.cv <- function(data, p, subset=NULL) {
  library("MASS")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- qda(y~., data=train, subset=subset)
    pred <- predict(fit, newdata=test[, -(p+1)], subset=subset)
    err[k] <- mean(pred$class == test$y)
    message(k)
  }
  return(err)
}

# err rate of naive-bayes using cross validation (non-nested)
naivebayes.cv <- function(data, p, subset=NULL) {
  library("naivebayes")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- naive_bayes(y~., data=train, subset=subset)
    pred <- predict(fit, newdata=test[, -(p+1)], subset=subset)
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}

multinom.cv <- function(data, p) {
  library("nnet")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- multinom(y~., data=train)
    pred <- predict(fit, newdata=test[, -(p+1)])
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}

# alpha = 0 : ridge
# alpha = 1 : lasso
lr.cv <- function(data, p, alpha) {
  library("glmnet")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  cv.out <- cv.glmnet(as.matrix(data[,-(p+1)]), data$y, 
                      type.measure="class", 
                      alpha=alpha, 
                      family="multinomial")
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- glmnet(as.matrix(train[,-(p+1)]), train$y, 
                        lambda=cv.out$lambda.min, 
                        alpha=alpha, 
                        family="multinomial")
    pred <- predict(fit, newx=as.matrix(test[, -(p+1)]), type="class")
    err[k] <- mean(pred == test$y)
  }
  return(err)
}

gam.cv <- function(data, p, nclass) {
  library("splines")
  library('nnet')
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
  fm <- paste('ns(', names(data)[1:p], ')', sep = "", collapse = ' + ')
  fm <- as.formula(paste('y~ ', fm))
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- multinom(formula=fm, data=train)
    pred <- predict(fit, newdata=test[, -(p+1)])
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}

rpart.cv <- function(data, p) {
  library("rpart")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- rpart(y~., data=train, 
                method = "class", 
                control = rpart.control(xval=10, minbucket=10, cp=0.00))
    pred <- predict(fit, newdata=test[, -(p+1)], type="class")
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}

prune.cv <- function(data, p) {
  library("rpart")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
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
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}

bagged.cv <- function(data, p) {
  library("randomForest")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- randomForest(y~., data=train, mtry=p)
    pred <- predict(fit, newdata=test[, -(p+1)], type="response")
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}

rf <- function(data, p) {
  library("randomForest")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- randomForest(y~., data=train)
    pred <- predict(fit, newdata=test[, -(p+1)], type="response")
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}

svm.c.cv <- function(data, p, kernel) {
  library("kernlab")
  CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
  N<-length(CC)
  M<-10 # nombre de répétitions de la validation croisée
  err<-matrix(0,N,M)
  x <- as.matrix(data[, -(p+1)])
  for(k in 1:M){
    for(i in 1:N){
      if (kernel == "vanilladot") {
        err[i,k]<-cross(ksvm(x=x, y=data$y, 
                             type="C-svc", 
                             kernel=kernel,
                             C=CC[i],
                             cross=5))
      }
      else if (kernel == "rbfdot") {
        err[i,k]<-cross(ksvm(x=x, y=data$y, 
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

svm.cv <- function(data, p, kernel, c) {
  library("kernlab")
  K <- 10
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- ksvm(x=as.matrix(train[, -(p+1)]), y=train$y, 
                type="C-svc", 
                kernel=kernel,
                C=c)
    pred <- predict(fit, newdata=test[, -(p+1)])
    err[k] <- mean(pred == test$y)
    message(k)
  }
  return(err)
}
