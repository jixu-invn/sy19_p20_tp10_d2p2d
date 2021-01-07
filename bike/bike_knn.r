source("../bike/bike.r")

raw_data = import_data_only_numeric()
X = get_train_test_and_validation_set(raw_data)

library(FNN)

best_k = 1
best_error = 99999999999999999
best_pred = NA
errors = c()
for (i in 1:100) {
  print(i)
  if(i == 2) #erreur bizarre lorsque k = 2
  {i = 3}
  knn = knn.reg(train = X$train_pred, test = X$test_pred, y = X$train_y, k = i)
  
  error = erreur_quadratique(knn$pred,X$test_y)
  errors = append(errors,error)
  if(error < best_error)
  {
    best_k = i
    best_error = error
    best_pred = knn$pred
  }
}

knn = knn.reg(train = X$train_pred, test = X$validation_pred, y = X$train_y, k = best_k)

error = erreur_quadratique(knn$pred,X$validation_y)

"best_k"
best_k
"erreur quadratique test"
best_error
"erreur quadratique validation"
error #483 209

plot_pred_test_val_vs_y(best_pred,knn$pred, as.matrix(X$validation_y))

errors = c()

for (i in 1:100) {
  X = get_train_and_test_set(raw_data)
  knn = knn.reg(train = X$train_pred, test = X$test_pred, y = X$train_y, k = best_k)
  errors = append(errors,erreur_quadratique(knn$pred,X$test_y))
}

errors_knn = errors
