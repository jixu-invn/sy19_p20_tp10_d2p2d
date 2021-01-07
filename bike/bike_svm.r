source("../bike/bike.r")
library(e1071)

raw_data = import_data()

X = get_train_and_test_set(raw_data)

X$train = subset (X$train, select = -yr)


svm_bike = svm(cnt ~ . , data=X$train, kernel="radial", gamma=1/9)
pred = predict(object = model,newdata = X$train_pred)

error = erreur_quadratique(pred,X$train_y)
error
X$test_y

save(svm_bike, file ='svm_bike.RData')
