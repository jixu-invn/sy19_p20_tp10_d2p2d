setwd("~/proj/sy19_p20_tp10_d2p2d")
#read
data <- read.table("data/letters_train.txt", header = T)
n.tot <- nrow(data)
n.dim <- ncol(data)

data.X <- data[,-1]
data.Y <- data[,1]

data <- cbind(data.X, data.Y)
names(data)[n.dim] <- 'y'
data$y <- as.factor(data$y)

n.class <- nlevels(data$y)

summary(data)
summary(data$y)


data.scale = as.data.frame(scale(data.X))
data.scale <- cbind(data.scale, data$y)
names(data.scale)[n.dim] <- 'y'










