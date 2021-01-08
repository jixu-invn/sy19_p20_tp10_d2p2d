setwd("~/proj/sy19_p20_tp10_d2p2d")
source("./letter/load_data.r")
source("./letter/cv.r")
lda.acc <- c(0,0,0,0,0,0,0,0,0,0)
qda.acc <- c(0,0,0,0,0,0,0,0,0,0)
nb.acc <- c(0,0,0,0,0,0,0,0,0,0)
multinom.acc <- c(0,0,0,0,0,0,0,0,0,0)
gam.acc <- c(0,0,0,0,0,0,0,0,0,0)
rpart.acc <- c(0,0,0,0,0,0,0,0,0,0)
prune.acc <- c(0,0,0,0,0,0,0,0,0,0)
bagged.acc <- c(0,0,0,0,0,0,0,0,0,0)
rf.acc <- c(0,0,0,0,0,0,0,0,0,0)
svm.linear.acc <- c(0,0,0,0,0,0,0,0,0,0)
svm.gaus.acc <- c(0,0,0,0,0,0,0,0,0,0)
nn.acc <- c(0,0,0,0,0,0,0,0,0,0)

#------LDA QDA NB--------#
lda.acc <- lda.cv(data, n.dim-1)
qda.acc <- qda.cv(data, n.dim-1)
nb.acc <- naivebayes.cv(data, n.dim-1)

#------Logistic linear regression--------#
multinom.acc <- multinom.cv(data, n.dim-1)

#------GAM------#
gam.acc <- gam.cv(data, n.dim-1, n.class)


#------Tree & Bagging------#
rpart.acc <- rpart.cv(data, n.dim-1)
prune.acc <- prune.cv(data, n.dim-1)

bagged.acc <- bagged.cv(data, n.dim-1)

rf.trees.acc <- rf.ntree.cv(data, n.dim-1)
rf.acc <- rf.cv(data, n.dim-1,1100)



#------SVM------#
c.min <- svm.c.cv(data, n.dim-1, kernel="vanilladot") # 1
svm.linear.acc <- svm.cv(data, n.dim-1, "vanilladot", 1)

c.min <- svm.c.cv(data, n.dim-1, kernel="rbfdot") # 100
svm.gaus.acc <- svm.cv(data, n.dim-1, "rbfdot", c = 100)

c.min <- svm.c.cv(data, n.dim-1, kernel="polydot") # 1
svm.poly.acc <- svm.cv(data, n.dim-1, "polydot", c = 1)



#------Neural Networks------#
nn.acc <- nn.cv(data,n.dim,n.class)



#------Display Results------#
boxplot(lda.acc, qda.acc, nb.acc,
        multinom.acc,
        gam.acc,
        rpart.acc, prune.acc, 
        bagged.acc, rf.acc,
        svm.linear.acc, svm.gaus.acc,
        nn.acc,
        names=c("lda", "qda", "nb", 
                "lr",
                "gam",
                "rpart", "prune",
                "bagged", "rf",
                "svm_l", "svm_g",
                "nn"))
#------Best methods------#
boxplot(rf.acc,
        svm.gaus.acc,
        names=c("rf",
                "svm_g"))





#------ Save Model------#
setwd("~/proj/sy19_p20_tp10_d2p2d/letter/")
l <- 390
data.test <- data[-390,]
data.try.X <- data[390,-17]
data.try.Y <- data[390,17]
data.read <- read.table("data/letters_train.txt", header = TRUE)

model.svm <- ksvm(x=as.matrix(data[, -n.dim]), y=data$y, 
                  type="C-svc", 
                  kernel="rbfdot",
                  kpar = "automatic",
                  C=100)
setwd("~/proj/sy19_p20_tp10_d2p2d/Livrable")
save(model.svm, file ='letter_svm.RData')



source("prediction.R")
pred <- prediction_letter(data.read[,-1])
print(pred)






