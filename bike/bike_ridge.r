source("../bike/bike.r")
library(glmnet)

raw_data = import_data_all_numeric()
raw_data = subset(x=raw_data,select=-c(yr))

errors = c()
for (i in 1:100) {
  X = get_train_and_test_set(raw_data)
  
  
  reg1 <- cv.glmnet(as.matrix(X$train_pred),as.matrix(X$train_y), family="gaussian", alpha=0)
  pred = predict(reg1,newx=as.matrix(X$test_pred))
  
  error = erreur_quadratique(pred,X$test_y)
  errors = append(errors,error)
}

errors_ridge = errors

errors = c()
for (i in 1:100) {
  X = get_train_and_test_set(raw_data)
  
  
  reg1 <- cv.glmnet(as.matrix(X$train_pred),as.matrix(X$train_y), family="gaussian", alpha=1)
  pred = predict(reg1,newx=as.matrix(X$test_pred))
  
  error = erreur_quadratique(pred,X$test_y)
  errors = append(errors,error)
}

errors_lasso = errors

errors = c()
for (i in 1:100) {
  X = get_train_and_test_set(raw_data)
  
  
  reg1 <- cv.glmnet(as.matrix(X$train_pred),as.matrix(X$train_y), family="gaussian")
  pred = predict(reg1,newx=as.matrix(X$test_pred))
  
  error = erreur_quadratique(pred,X$test_y)
  errors = append(errors,error)
}

errors_elastic = errors

errors = data.frame(errors_lasso,errors_ridge,errors_elastic)
boxplot(errors)
