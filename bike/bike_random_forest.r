source("./bike.r")
library(randomForest)

raw_data = import_data()
X = get_train_and_test_set(raw_data)

f = randomForest(cnt ~ ., 
             data = X$train, na.action = na.omit)


summary(f)
plot(f)
opt = which.min(f$mse)
opt

pred = predict(object = f,newdata = X$test_pred)

erreur_quadratique(pred,X$test_y) #215 767

plot_pred_vs_y(pred,as.matrix(X$test_y))


f = randomForest(cnt ~ ., 
                 data = X$train, na.action = na.omit, ntree=opt) 

f$mse

pred = predict(object = f,newdata = X$test_pred)

erreur_quadratique(pred,X$test_y) #219 965

plot_pred_vs_y(pred,as.matrix(X$test_y))
