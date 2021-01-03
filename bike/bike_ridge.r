source("./bike.r")
library(glmnet)

raw_data = import_data_only_numeric()
X = get_train_and_test_set(raw_data)

reg1 <-glm(cnt~., data =X$train, family ="binomial")
pred = predict(reg1,newx=X$test_pred)

  knn = knn.reg(train = X$train_pred, test = X$test_pred, y = X$train_y, k = i)

error = erreur_quadratique(knn$pred,X$validation_y)

"best_k"
best_k
"erreur quadratique test"
best_error
"erreur quadratique validation"
error

plot_pred_test_val_vs_y(best_pred,knn$pred, as.matrix(X$validation_y))

