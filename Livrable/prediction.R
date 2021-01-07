prediction_phoneme <- function(dataset) {
  # Chargement de l’environnement
  library(tensorflow)
  library(keras)
  load("phoneme.Rdata") # load res.pca
  model.serialize <- serialize_model(load_model_hdf5("phoneme.h5"))
  model <- unserialize_model(model.serialize)
  
  classes <- c("aa", "ao", "dcl","iy", "sh") 
  
  # Transformation ACP
  nb.p <- ncol(dataset)
  dataset.pca <- as.data.frame((scale(dataset[, -257], center=res.pca$center) %*% res.pca$rotation)[, 1:res.pca$n])
  
  # Prédiction
  pred <- predict(model, as.matrix(dataset.pca))
  predictions <- rep(0, nrow(dataset))
  for (i in 1:nrow(dataset)) {
    predictions[i] <- classes[which.max(pred[i,])]
  }
  return(as.factor(predictions))
}

prediction_letter <- function(dataset) {
  # Chargement de l’environnement
  load("env.Rdata")
  # Mon algorithme qui renvoie les prédictions sur le jeu de données
  # ‘dataset‘ fourni en argument.
  # ...
  return(predictions)
}

prediction_bike <- function(list) {
  # Chargement de l environnement
  load("svm_bike.Rdata")
  predictions = predict(object = svm_bike,newdata = list)
  return(predictions)
}