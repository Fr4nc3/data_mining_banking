## UNUSED MODEL that failed


######################################################
### Regression Tree ###
######################################################
### Train model ###
mod <- train(Y_AcceptedOffer ~ .,
  data = trainData,
  method = "rpart"
)


### Get predictions ###
ptest <- predict(mod, newdata = testData)
ptrain <- predict(mod, newdata = trainData)

### Accuracy ###

# Accuracy for training set
mean(ptrain == trainData$Y_AcceptedOffer)

# Accuracy for Test set
mean(ptest == testData$Y_AcceptedOffer)

### Regression Tree END ###
######################################################


######################################################
### Logistic Regression ###
######################################################
### Train model ###
mod <- train(Y_AcceptedOffer ~ .,
  data = trainData,
  method = "glm",
  family = "binomial"
)


### Get predictions ###
ptest <- predict(mod, newdata = testData)
ptrain <- predict(mod, newdata = trainData)

### Accuracy ###

# Accuracy for training set
mean(ptrain == trainData$Y_AcceptedOffer)

# Accuracy for Test set
mean(ptest == testData$Y_AcceptedOffer)

### Logistic Regression END ###
######################################################


######################################################
### Partial Least Squares ###
######################################################
### Train model ###
mod <- train(Y_AcceptedOffer ~ .,
  data = trainData,
  method = "pls",
  preProc = c("center", "scale")
)


### Get predictions ###
ptest <- predict(mod, newdata = testData)
ptrain <- predict(mod, newdata = trainData)

### Accuracy ###

# Accuracy for training set
mean(ptrain == trainData$Y_AcceptedOffer)

# Accuracy for Test set
mean(ptest == testData$Y_AcceptedOffer)

### Logistic Regression END ###
######################################################



trainClasses_glm <- ifelse(trainingPreds_glm >= cutoff, 1, 0)
trainClasses_glm
# Organize w/Actual
trainResults_glm <- data.frame(
  actual = treatedTrain$Y_AcceptedOffer,
  probablity = trainingPreds_glm,
  classes = trainClasses_glm
)
head(trainResults_glm)

# Get a confusion matrix
(confMat <- ConfusionMatrix(trainResults_glm$classes, trainResults_glm$actual))
Accuracy(trainResults_glm$classes, trainResults_glm$actual)

testClasses_glm <- ifelse(testingPreds_glm >= cutoff, 1, 0)

# Organize w/Actual
testResults_glm <- data.frame(treatedTest$HHuniqueID,
  actual = treatedTest$Y_AcceptedOffer,
  probablity = testingPreds_glm,
  classes = testClasses_glm
)
head(testResults_glm)

# Get a confusion matrix
(confMat <- ConfusionMatrix(testResults_glm$classes, testResults_glm$actual))
Accuracy(testResults_glm$classes, testResults_glm$actual)