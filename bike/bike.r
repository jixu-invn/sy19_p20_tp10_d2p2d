instant_present = FALSE

import_data <- function() {
  X = read.csv("../data/bike_train.csv", sep=',')
  
  X$dteday = as.Date(X$dteday,format = "%Y-%m-%d") #on cast les dates en dates
  X$season = as.factor(X$season) #on cast les valeurs discrètes en facteur
  X$yr = as.factor(X$yr) #on cast les valeurs discrètes en facteur
  X$mnth = as.factor(X$mnth) #on cast les valeurs discrètes en facteur
  X$holiday = as.factor(X$holiday) #on cast les valeurs discrètes en facteur
  X$weekday = as.factor(X$weekday) #on cast les valeurs discrètes en facteur
  X$workingday = as.factor(X$workingday) #on cast les valeurs discrètes en facteur
  X$weathersit = as.factor(X$weathersit) #on cast les valeurs discrètes en facteur
  
  if(!instant_present){
    X = subset(x = X, select=-instant) #on retire l'index de la mesure parce qu'elle n'est pas signifiante
  }
  
  X
}

import_data_all_numeric <- function() { # on garde les prédicteurs factoriels mais sous forme numérique
  X = import_data()
  X = subset(x = X, select=-cd(dteday))
  X
}

import_data_only_numeric <- function() { # on en enlève les prédicteurs factoriels
  X = import_data()
  X = subset(x = X, select=-c(dteday,season,yr,mnth,holiday,weekday,workingday,weathersit))
  X
  }


get_train_and_test_set <- function(X) {
  library(caret)
  
  set.seed(135)
  
  training.samples <- createDataPartition(p = 0.8, list = FALSE, y = X$cnt)
  train = X[training.samples,]
  test = X[-training.samples,]
  
  list(train=train,train_pred=subset(train,select=-cnt),train_y=subset(train,select=cnt),
       test=test,test_pred=subset(test,select=-cnt),test_y=subset(test,select=cnt))
}

get_train_test_and_validation_set <- function(X) {
  library(caret)
  
  set.seed(135)
  
  training.samples <- createDataPartition(p = 0.6, list = FALSE, y = X$cnt)
  train = X[training.samples,]
  other = X[-training.samples,]
  testing.samples <- createDataPartition(p = 0.5, list = FALSE, y = other$cnt)
  test = other[testing.samples,]
  validation = other[-testing.samples,]
  
  list(train=train,train_pred=subset(train,select=-cnt),train_y=subset(train,select=cnt),
       test=test,test_pred=subset(test,select=-cnt),test_y=subset(test,select=cnt),
       validation=validation,validation_pred=subset(validation,select=-cnt),validation_y=subset(validation,select=cnt))
}

erreur_quadratique <- function(pred,y) {
  sum((pred-y)^2)/length(pred)
}

plot_pred_vs_y <- function(pred,y) {
  plot(y,col="blue")
  points(pred,col="red")
}

plot_pred_test_val_vs_y <- function(pred_test,pred_val,y) {
  plot(y,col="blue")
  points(pred_test,col="red")
  points(pred_val,col="green")
}

