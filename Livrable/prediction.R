prediction_phoneme <- function(dataset) {
  # Chargement de l’environnement
  library(tensorflow)
  library(keras)
  load("env.Rdata")
  model.serialize <- serialize_model(load_model_hdf5("phoneme.h5"))
  model <- unserialize_model(model.serialize)
  
  classes <- c("aa", "ao", "dcl","iy", "sh") 
  
  # Transformation ACP
  nb.p <- ncol(dataset)
  dataset.pca <- as.data.frame((scale(dataset[, -257], center=phoneme.pca$center) %*% phoneme.pca$rotation)[, 1:phoneme.pca$n])
  
  # Prédiction
  pred <- predict(model, as.matrix(dataset.pca))
  predictions <- rep(0, nrow(dataset))
  for (i in 1:nrow(dataset)) {
    predictions[i] <- classes[which.max(pred[i,])]
  }
  return(as.factor(predictions))
}

prediction_letter <- function(dataset) {
  load("env.RData")
  library("kernlab")
  if (ncol(dataset) == 17) {
    newdata <- dataset[, -1]
  }
  else {
    newdata <- dataset
  }
  predictions = predict(letter.svm, newdata = newdata)
  return(predictions)
}

prediction_bike <- function(dataset) {
  # Chargement de l environnement
  load("env.Rdata")
  library("e1071")
  
  # Transformation
  dataset$dteday = as.Date(dataset$dteday,format = "%Y-%m-%d") 
  dataset$season = as.factor(dataset$season) 
  dataset$yr = as.factor(dataset$yr) 
  dataset$mnth = as.factor(dataset$mnth) 
  dataset$holiday = as.factor(dataset$holiday) 
  dataset$weekday = as.factor(dataset$weekday) 
  dataset$workingday = as.factor(dataset$workingday) 
  dataset$weathersit = as.factor(dataset$weathersit) 
  
  predictions = predict(object = bike.svm, newdata = dataset[, -14])
  return(predictions)
}