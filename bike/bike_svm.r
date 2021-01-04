source("./bike.r")
library(e1071)

raw_data = import_data()
X = get_train_and_test_set(raw_data)

X$train = subset (X$train, select = -yr)

model = svm(cnt ~ . , data=X$train)

summary(model)

pred = predict(object = model,newdata = X$test_pred)

erreur_quadratique(pred,X$test_y) # 286 313
plot_pred_vs_y(pred,as.matrix(X$test_y))
