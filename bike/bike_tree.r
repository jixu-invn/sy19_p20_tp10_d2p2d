source("./bike.r")
library(tree)

raw_data = import_data()
X = get_train_and_test_set(raw_data)


t = tree(cnt~.,data=X$train)

summary(t)
plot(t)

pred = predict(object = t,newdata = X$test_pred)

erreur_quadratique(pred,X$test_y) # 367 792 sans kernel, 341 946 avec kernel sqrt
plot_pred_vs_y(pred,as.matrix(X$test_y))

library(rpart)

t = rpart(cnt~.,data=X$train)

summary(t)
plot(t)

pred = predict(object = t,newdata = X$test_pred)

erreur_quadratique(pred,X$test_y) #516 744
plot_pred_vs_y(pred,as.matrix(X$test_y))
