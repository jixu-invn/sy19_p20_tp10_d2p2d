X = read.csv("D:/UTC/Semestre 9/SY19/TP10/sy19_p20_tp10_d2p2d/data/bike_train.csv", sep=',')

X$dteday = as.Date(X$dteday,format = "%Y-%m-%d") #on cast les dates en dates
X$season = as.factor(X$season) #on cast les valeurs discrètes en facteur
X$yr = as.factor(X$yr) #on cast les valeurs discrètes en facteur
X$mnth = as.factor(X$mnth) #on cast les valeurs discrètes en facteur
X$holiday = as.factor(X$holiday) #on cast les valeurs discrètes en facteur
X$weekday = as.factor(X$weekday) #on cast les valeurs discrètes en facteur
X$workingday = as.factor(X$workingday) #on cast les valeurs discrètes en facteur
X$weathersit = as.factor(X$weathersit) #on cast les valeurs discrètes en facteur

#plot(cnt~dteday, data=X) 

predicteurs = X[,3:13] #on récupère les prédicteurs uniquement
predicteurs_numeriques = X[,10:14]
names(predicteurs_numeriques)[names(predicteurs_numeriques)=="cnt"] <- "y" #on renomme la colomne objective en y


ACP = prcomp(as.matrix(predicteurs_numeriques)) #on réalise une ACP puisque les prédicteurs semblent hautement corrélés

summary(ACP) 

plot(ACP$sdev) # la méthode du coude suggère de garder 1 composante

ACP = prcomp(as.matrix(predicteurs_numeriques), rank.=1, retx=TRUE) 

ACP_X = data.frame(ACP$x)
ACP_X$y = X$cnt
#ACP_X contient les données après ACP

## Création de données d'apprentissage et de test
library(caret)

training.samples <- createDataPartition(p = 0.6, list = FALSE, y = ACP_X$y)
train_ACP = ACP_X[training.samples,]
test_ACP = ACP_X[-training.samples,]

train = predicteurs_numeriques[training.samples,]
test = predicteurs_numeriques[-training.samples,]

## régression KNN pour tester

library(FNN)

knn = knn.reg(train = as.data.frame(train_ACP[,1]), test = as.data.frame(test_ACP[,1]), y = train_ACP$y, k = 1)

erreur_quadratique1 = sqrt(sum((knn$pred-test_ACP$y)^2)/length(knn$pred))
erreur_quadratique1
plot(test_ACP$y,col="blue")
points(best_knn$pred,col="red")

"""knn = knn.reg(train = as.data.frame(train[,1:3]), test = as.data.frame(test[,1:3]), y = train$y, k = 20)

erreur_quadratique2 = sqrt(sum((knn$pred-test$y)^2)/length(knn$pred))
erreur_quadratique2
plot(test$y,col="blue")
points(knn$pred,col="red")

mean(X$cnt)"""
