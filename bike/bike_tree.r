source("./bike.r")
library(tree)

raw_data = import_data()

tree_errors = c()
for (i in 1:100) {
  X = get_train_and_test_set(raw_data)
  
  t = tree(cnt~.,data=X$train)
  
  pred = predict(object = t,newdata = X$test_pred)
  
  tree_errors = append(tree_errors,erreur_quadratique(pred,X$test_y)) # 367 792 sans kernel, 341 946 avec kernel sqrt
}

t_er = data.frame(tree_errors)
boxplot(t_er)

