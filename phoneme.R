#------Read Data--------#
data <- read.table("data/phoneme_train.txt", header = TRUE)
data$y <- as.factor(data$y)
summary(data$y)
nb.p <- ncol(data)-1
nb.class <- nlevels(data$y)

#------Split Train/Test Set--------# # use cross-validation instead
#library(caret)
#set.seed(1729)
#train.index <- createDataPartition(y=data$y, p=0.7, list=FALSE)
#train <- data[train.index,]
#test <- data[-train.index,]

#------Principle component analysis--------#
#res.pca <- prcomp(x=train[, -(nb.p+1)], center=TRUE, scale.=TRUE)
#plot(cumsum(res.pca$sdev^2 / sum(res.pca$sdev^2)), type="l", ylim=0:1) # plotting cumulative proportion
#summary(res.pca)

res.pca <- prcomp(x=data[, -(nb.p+1)], center=TRUE, scale.=TRUE)
#plot(cumsum(res.pca$sdev^2 / sum(res.pca$sdev^2)), type="l", ylim=0:1)
summary(res.pca)

nb.pca <- 128 # we take n first principal components

#train.pca <- as.data.frame(res.pca$x[, 1:nb.pca]) 
#train.pca <- cbind(train.pca, train$y)
#names(train.pca)[nb.pca+1] <- 'y'
#test.pca <- as.data.frame((scale(test[, -(nb.p+1)], center=res.pca$center) %*% res.pca$rotation)[, 1:nb.pca])
#test.pca <- cbind(test.pca, test$y)
#names(test.pca)[nb.pca+1] <- 'y'
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, data$y)
names(data.pca)[nb.pca+1] <- 'y'

#------LDA QDA NB--------#
source("cv.R")
lda.err <- lda.cv(data.pca, nb.pca)
qda.err <- qda.cv(data.pca, nb.pca)
nb.err <- naivebayes.cv(data.pca, nb.pca)

#------Logistic linear regression--------#
multinom.err <- multinom.cv(data.pca, nb.pca)
ridge.err <- lr.cv(data.pca, nb.pca, 0)
lasso.err <- lr.cv(data.pca, nb.pca, 1)

#------GAM------#
gam.err <- gam.cv(data.pca, nb.pca, nb.class)

#------Tree & Bagging------#
rpart.err <- rpart.cv(data.pca, nb.pca)
prune.err <- prune.cv(data.pca, nb.pca)

bagged.err <- bagged.cv(data.pca, nb.pca)
rf.err <- rf.cv(data.pca, nb.pca)

#------SVM------#
svm.linear.c <- svm.c.cv(data.pca, nb.pca, kernel="vanilladot") # 0.01
svm.linear.err <- svm.cv(data.pca, nb.pca, "vanilladot", 0.01)

svm.gaus.c <- svm.c.cv(data.pca, nb.pca, kernel="rbfdot") # 1
svm.gaus.err <- svm.cv(data.pca, nb.pca, "rbfdot", 1)

#------Neural Networks------#
source("nn.R")
ann.err <- ann.cv(data.pca, nb.pca, nb.class)
dnn.err <- dnn.cv(data.pca, nb.pca, nb.class)
cnn.err <- cnn.cv(data.pca, nb.pca, nb.class, 8)

# Final model
model <- keras_model_sequential()
model %>% layer_dense(units=nb.pca, activation="relu", input_shape=nb.pca) %>%
        layer_dense(units=32, activation="relu") %>%
        layer_dropout(0.5) %>%
        layer_dense(units=16, activation="relu") %>%
        layer_dropout(0.5) %>%
        layer_dense(units=nb.class, activation="softmax")
model %>% compile(loss="categorical_crossentropy", optimizer='adam', metrics='accuracy')
model %>% fit(as.matrix(data.pca[,-(nb.pca+1)]), model.matrix(~ -1 + y, data=data.pca),
              epochs=50, batch_size=128, validation_split=0)
model %>% evaluate(as.matrix(data.pca[,-(nb.pca+1)]), model.matrix(~ -1 + y, data=data.pca))

# Save model #
model %>% save_model_hdf5("Livrable/phoneme.h5")
model.serialize <- serialize_model(load_model_hdf5("keras_out.h5"))
model <- unserialize_model(model.serialize)

# test
setwd("~/SY19_Workspace/sy19_p20_tp10_d2p2d")
data <- read.table("data/phoneme_train.txt", header = TRUE)
setwd("~/SY19_Workspace/sy19_p20_tp10_d2p2d/Livrable")
source("prediction.R")
pred <- prediction_phoneme(data)
mean(pred == data$y)
