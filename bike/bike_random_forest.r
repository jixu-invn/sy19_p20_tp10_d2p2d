source("./bike.r")
library(randomForest)

raw_data = import_data()
forest_errors = c()

for (i in 1:100) {

  X = get_train_and_test_set(raw_data)
  
  f = randomForest(cnt ~ ., 
               data = X$train, na.action = na.omit)
  
  
  opt = which.min(f$mse)
  
  pred = predict(object = f,newdata = X$test_pred)
  
  forest_errors = append(forest_error,erreur_quadratique(pred,X$test_y)) #215 767
}

forest_errors = forest_errors[1:100]
errors_forest = forest_errors
t_error = data.frame(tree_errors,forest_errors)
errors_tree = tree_errors
boxplot(t_error)
