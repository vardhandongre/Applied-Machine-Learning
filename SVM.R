# Training and Testing a SVM 
setwd("/Users/don/Desktop/AML")
library(caret)

############################################ Parameters ###########################################
lambdas <- c(0.001, .01, .1, 1)
num_epochs <- 50
num_steps <- 300
steps_till_eval = 30

## Parameters for LR
n = 50
m = 0.01 

########################################### Dataset ###########################################
train_fraction <- 0.8
test_fraction <- 0.1
valid_fraction <- 0.1 

wtd1 <- read.csv('adult.data.txt', header = FALSE, na.strings = "?")
wtd2 <- read.csv('adult.test.txt', header = FALSE, na.strings = "?")
dtset <- rbind(wtd1, wtd2, make.row.names = FALSE) 
## Binary labels ( >50K : +1 , <50K : -1)
#dtset$income <- ifelse(dtset[,15]==" >50K" | dtset[,15]==" 50K.", 1, -1)
#as.factor(dtset$income)
# Seperate continuous features
# c(1,3,5,11,12,13)

dtset_X <- dtset[,c(1,3,5,11,12,13)]
dtset_Y <- dtset[,15]

########################################### Scaling data ###########################################
for (iter in 1:dim(dtset_X)[2]){
  dtset_X[iter] <- scale(as.numeric(as.matrix(dtset_X[iter])))
}

########################################### Dataset splitting ###########################################
split_index <- createDataPartition(y = dtset_Y, p = train_fraction, list = FALSE) 
train_X <- dtset_X[split_index, ]
train_Y <- dtset_Y[split_index]

########################################### Make a 50-50 split in the remaining data ##########################################

remaining_data_X <- dtset_X[-split_index, ] 
remaining_data_Y <- dtset_Y[-split_index]  

split_index2 <- createDataPartition(y = remaining_data_Y, p = 0.5, list = FALSE) 
valid_X <- remaining_data_X[split_index2, ]
valid_Y <- remaining_data_Y[split_index2]
test_X <- remaining_data_X[-split_index2, ]
test_Y <- remaining_data_Y[-split_index2]

########################################## Helper Functions ##########################################
# convert2Label1 (for actual dataset)
# convert2Label2 (for predicted values)
# accuracy
# classifier ( t(a)%*%x + b )
# Examples for reference [ ex1, ex2 for +1 and ex3 ex4 for -1]
ex1 <- dtset_Y[10]
ex2 <- dtset_Y[48843]
ex3 <- dtset_Y[6]
ex4 <- dtset_Y[48842]

convert2Label1 <- function(y){
  if(y == ex1 | y == ex2){
    return (1)
  } 
  
  else if(y == ex3 | y == ex4){
    return (-1)
  }
}

convert2Label2 <- function(y){
  if(y >= 0){
    return (1)
  }
  else{
    return(-1)
  }
}

classifier <- function(x, a, b){
  x_n <- as.numeric(as.matrix(x))
  return (t(a) %*% x_n + b)
}

accuracy <- function(x,y,a,b){
  correct <- 0
  wrong <- 0
  for (i in 1:length(y)){
    pred <- classifier(x[i,], a, b)
    pred <- convert2Label2(pred)
    actual <- convert2Label1(y[i])
    
    if(pred == actual){
      correct <- correct + 1 
    } else{
      wrong <- wrong + 1
    }
  }
  return(c( (correct/(correct+wrong))) )
}

########################################### Training ###########################################
val_acc <- c()
test_acc <- c()

for (lm in lambdas){
  # initialize parameters
  a <- rep(0,6)
  b <- 0
  accuracy_step = c()
  
  for (e in 1:num_epochs){
    
    # Seperate 50 examples for testing every 30 steps
    eval_indices <- sample(1:dim(train_X)[1], 50)
    eval_X <- train_X[eval_indices, ] ############################################################# Consider only for 30th step evaluation
    eval_Y <- train_Y[eval_indices] ############################################################# Consider only for 30th step evaluation
    train_data_x <- train_X[-eval_indices, ] ############################################################# Select samples from this for SGD
    train_data_y <- train_Y[-eval_indices] ############################################################# Select sample from this for SGD
    
    step_count <- 0
    
    for (s in 1:num_steps){
      
      if(step_count %% steps_till_eval == 0){
        find_acc <- accuracy(eval_X, eval_Y, a, b)
        accuracy_step <- c(accuracy_step, find_acc[1]) 
        #print(calc[1])
      }
      
      # Data Sample for performing SGD
      #sample_index <- sample(1:length(train_data_y), 1)  # Batch Size Nb = 1
      #while(is.na( convert2Label1( train_data_y[sample_index] ))){
      #  sample_index <- sample(1:length(train_data_y), 1)
      #}
      
      sample_index <- sample(1:length(train_data_y), 1)
      while(is.null( convert2Label1( train_data_y[sample_index] ) )){
        sample_index <- sample(1:length(train_data_y), 1)
      }
      x_data <- as.numeric(as.matrix(train_data_x[sample_index, ]))
      y_data <- convert2Label1(train_data_y[sample_index])
      
      # Predict using the a and b values
      pred <- classifier(x_data, a, b)
      step_len <- 1 / ((m*e)+n)
      
      # Update Rule for the parameters
      if (y_data*pred >= 1){
        da <- lm * a 
        db <- 0
      } else {
        da <- (lm * a) - (y_data*x_data)
        db <- -1*y_data
      }
      
      a <- a - step_len*da
      b <- b - step_len*db
      
      step_count <- step_count + 1
    }
  }
  
  # Validation
  val_eval <- accuracy(valid_X, valid_Y, a, b)
  val_acc <- c(val_acc, val_eval[1])
  
  # Testing
  test_eval <- accuracy(test_X, test_Y, a, b)
  test_acc <- c(test_acc, test_eval[1])
  
  # Plotting ( Ref: stackoverflow/R documentation) [https://stackoverflow.com/questions/7144118/how-to-save-a-plot-as-image-on-the-disk]
  jpeg(file = paste(toString(lm), ".jpg"))
  title <- paste("lm = ", toString(lm), "Accuracy")
  plot(1:length(accuracy_step), accuracy_step, type = 'o', col = "green", xlab = "Time", ylab ="Accuracy", main = title)
  dev.off()
}

########################################## Part b ##########################################
max <- 1

for (i in 1:length(val_acc)) {
  if(val_acc[i]>val_acc[max]){
    max <- i
  }
}

# Best Lambda
lambdas[max]

########################################## Part c ##########################################

test_acc[max]



