######################################################
# Francia F. Riesco
# Banking Modeling
######################################################
# This R file run after banking_eda.R, ie, this needs the variables from that file
# Examine the levels of Y
levels(as.factor(joinData$Y_AcceptedOffer))
######################################################
# This is a classification problem so ensure R knows Y isn't 0/1 as integers
######################################################
joinData$Y_AcceptedOffer <- as.factor(joinData$Y_AcceptedOffer)


######################################################
# SAMPLE: Partition schema
# Partitioning 20% test set
######################################################
splitPercent <- round(nrow(joinData) %*% .8)

set.seed(1234)
idx <- sample(1:nrow(joinData), splitPercent)
trainData <- joinData[idx, ]
testData <- joinData[-idx, ]

dim(trainData)
dim(testData)

is.factor(joinData$Y_AcceptedOffer)

xVars <- c(
  "Communication", "LastContactDay", "LastContactMonth", "NoOfContacts",
  "DaysPassed", "PrevAttempts", "past_Outcome", "carMake", "carModel", "carYr",
  "DefaultOnRecord", "RecentBalance", "HHInsurance", "CarLoan",
  "headOfhouseholdGender", "PetsPurchases", "DigitalHabits_5_AlwaysOn",
  "AffluencePurchases", "Age", "Job", "Marital", "Education", "CallMin"
)
y <- "Y_AcceptedOffer"
successClass <- "Accepted"
######################################################
# Testing Generalized Linear Models
######################################################
model <- glm(I(Y_AcceptedOffer == "Accepted") ~ Age + Communication + Marital + Education + Job + headOfhouseholdGender
  + CallMin + carYr + NoOfContacts + carMake + carModel + DaysPassed + PrevAttempts + CarLoan + PetsPurchases
  + AffluencePurchases + DefaultOnRecord + RecentBalance, data = joinData, family = binomial())
summary(model)

step(model, test = "LRT")

# model suggested by step
model <- glm(formula = I(Y_AcceptedOffer == "Accepted") ~ Age + Communication + Marital +
  Education + Job + headOfhouseholdGender + CallMin + carYr +
  NoOfContacts + carMake + carModel + DaysPassed + PrevAttempts +
  CarLoan + PetsPurchases + AffluencePurchases + DefaultOnRecord +
  RecentBalance, family = binomial(), data = joinData)
summary(model)


######################################################
## MODIFY: Vtreat, need to declare xVars & name of Y var
######################################################

plan <- designTreatmentsC(
  joinData,
  xVars,
  y,
  successClass
)
treatedData <- prepare(plan, joinData)
fit <- glm(as.factor(Y_AcceptedOffer) ~ ., treatedData, family = "binomial")

summary(fit)

# Make real predictions
probabilitAccept <- predict(fit, treatedData, type = "response")

head(probabilitAccept)
plot(density(probabilitAccept))


step(fit, test = "LRT")
# step suggested this as the best model
fit_by_step <- glm(
  formula = as.factor(Y_AcceptedOffer) ~ NoOfContacts + PrevAttempts +
    past_Outcome_catB + carModel_catB + HHInsurance + CarLoan +
    PetsPurchases + Job_catB + Marital_catB + Education_catB +
    CallMin + Communication_lev_x_cellular + LastContactMonth_lev_x_apr +
    LastContactMonth_lev_x_aug + LastContactMonth_lev_x_feb +
    LastContactMonth_lev_x_jan + LastContactMonth_lev_x_jul +
    LastContactMonth_lev_x_jun + LastContactMonth_lev_x_may +
    LastContactMonth_lev_x_nov + Job_lev_x_management, family = "binomial",
  data = treatedData
)

summary(fit_by_step)

# Make real predictions
probabilitAccept <- predict(fit_by_step, treatedData, type = "response")

head(probabilitAccept)
par(mar = c(1, 1, 1, 1))
plot(density(probabilitAccept))

######################################################
## Data Modeling
######################################################
# plan <- designTreatmentsC(..., xVars, ..., 1)
plan <- designTreatmentsC(
  trainData,
  xVars,
  y,
  successClass
)

# Apply the rules to the set
treatedTrain <- prepare(plan, trainData)
treatedTest <- prepare(plan, testData)
######################################################
# 3. Apply a treatment plan
######################################################
treatedProspects <- prepare(plan, joinProspects)


## MODEL: caret etc.
######################################################
# Fit lm model using 10-fold CV: model
######################################################
crtl <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE
)

set.seed(1234)
fit_glm <- train(Y_AcceptedOffer ~ .,
  data = treatedTrain, method = "glm",
  family = "binomial",
  trControl = crtl
)
# saveRDS(fit_glm , 'glm_finalFit_10Fold.rds')
fit_glm <- readRDS("glm_finalFit_10Fold.rds")
## ASSESS: Predict & calculate the KPI appropriate for classification
trainingPreds_glm <- predict(fit_glm, treatedTrain)
testingPreds_glm <- predict(fit_glm, treatedTest)
tail(trainingPreds_glm)
### Accuracy ###

# Accuracy for training set
mean(trainingPreds_glm == treatedTrain$Y_AcceptedOffer) # [1] 0.8696875

# Accuracy for Test set
mean(testingPreds_glm == treatedTest$Y_AcceptedOffer) # [1] 0.7375

# Classify
cutoff <- 0.8

####################
trainingPreds_glm <- predict(fit_glm, treatedTrain)
trainClasses_glm <- ifelse(trainingPreds_glm >= cutoff, 1, 0)
trainingPreds_glm
# Organize w/Actual
trainResults_glm <- data.frame(
  actual = treatedTrain$Y_AcceptedOffer,
  probablity = trainingPreds_glm,
  classes = trainClasses_glm
)
head(trainResults_glm)

# Get a confusion matrix
confusionMatrix(trainingPreds_glm, treatedTrain$Y_AcceptedOffer)
####################


# 4. Make predictions
prospectPreds_glm <- predict(fit_glm, treatedProspects, type = "prob")

prospectClasses_glm <- ifelse(prospectPreds_glm >= cutoff, 1, 0)
tail(prospectClasses_glm)
# 5. Join probabilities back to ID
prospectsResults_glm <- data.frame(joinProspects$HHuniqueID,
  probablity = prospectPreds_glm
)
head(prospectsResults_glm, 100)

# 6. Identify the top 100 "success" class probabilities from prospectsResults
head(prospectsResults_glm, 100)
prospectsResults_glm_sorted <- prospectsResults_glm %>% arrange(desc(probablity.Accepted))
head(prospectsResults_glm_sorted, 100)



######################################################
# Model k-Nearest neighbor KNN
######################################################

knnFit <- train(Y_AcceptedOffer ~ ., # similar formula ln
  data = treatedTrain, # data input
  method = "knn", # caret has other methods so specify KNN
  preProcess = c("center", "scale")
) # normalization
knnFit
plot(knnFit)
saveRDS(knnFit, "knnFitfullData.rds")
knnFit <- readRDS("knnFitfullData.rds")

trainingPreds_knn <- predict(knnFit, treatedTrain)
testingPreds_knn <- predict(knnFit, treatedTest)
# Accuracy for training set
mean(trainingPreds_knn == treatedTrain$Y_AcceptedOffer) # 0.773125

# Accuracy for Test set
mean(testingPreds_knn == treatedTest$Y_AcceptedOffer) # 0.695

confusionMatrix(trainingPreds_knn, treatedTrain$Y_AcceptedOffer)
confusionMatrix(testingPreds_knn, treatedTest$Y_AcceptedOffer)

# 4. Make predictions
prospectPreds_knn <- predict(knnFit, treatedProspects, type = "prob")

prospectClasses_knn <- ifelse(prospectPreds_knn >= cutoff, 1, 0)
tail(prospectClasses_knn)
# 5. Join probabilities back to ID
prospectsResults_knn <- data.frame(joinProspects$HHuniqueID,
  probablity = prospectPreds_knn
)

# 6. Identify the top 100 "success" class probabilities from prospectsResults
head(prospectsResults_knn, 100)
prospectsResults_knn_sorted <- prospectsResults_knn %>% arrange(desc(probablity.Accepted))
head(prospectsResults_knn_sorted, 100)


######################################################
# Model   caret does recursive partitioning (trees)
######################################################

set.seed(1234)
fit_rpart <- train(Y_AcceptedOffer ~ .,
  data = treatedTrain, # data input
  method = "rpart",
  tuneGrid = data.frame(cp = c(0.1, 0.01, 0.05, 0.07)),
  control = rpart.control(minsplit = 1, minbucket = 2)
)
# saveRDS(fit_rpart,'fullDataTreeFit.rds')
fit_rpart <- readRDS("fullDataTreeFit.rds")
plot(fit_rpart)
prp(fit_rpart$finalModel, extra = 1)

# Make some predictions on the training set

trainingPreds_rpart <- predict(fit_rpart, treatedTrain)
testingPreds_rpart <- predict(fit_rpart, treatedTest)
# Accuracy for training set
mean(trainingPreds_rpart == treatedTrain$Y_AcceptedOffer) #  0.8471875

# Accuracy for Test set
mean(testingPreds_rpart == treatedTest$Y_AcceptedOffer) # 0.725
# Get the conf Matrix
confusionMatrix(trainingPreds_rpart, treatedTrain$Y_AcceptedOffer)

# 4. Make predictions
prospectPreds_rpart <- predict(fit_rpart, treatedProspects, type = "prob")

prospectClasses_rpart <- ifelse(prospectPreds_rpart >= cutoff, 1, 0)
tail(prospectClasses_rpart)
# 5. Join probabilities back to ID
prospectsResults_rpart <- data.frame(joinProspects$HHuniqueID,
  probablity = prospectPreds_rpart
)
head(prospectsResults_rpart, 100)

# 6. Identify the top 100 "success" class probabilities from prospectsResults
head(prospectsResults_rpart, 100)
prospectsResults_rpart_sorted <- prospectsResults_rpart %>% arrange(desc(probablity.Accepted))
head(prospectsResults_rpart_sorted, 100)


######################################################
# Model random forest model with Caret
######################################################
set.seed(1234)
fit_rf <- train(Y_AcceptedOffer ~ .,
  data = treatedTrain, # data input
  method = "rf",
  verbose = FALSE,
  ntree = 3,
  tuneGrid = data.frame(mtry = 1)
) # num of vars used in each tree
fit_rf

saveRDS(fit_rf, "fullDataRandomFR.rds")
fit_rf <- readRDS("ullDataRandomFR.rds")
trainingPreds_rf <- predict(fit_rf, treatedTrain)
testingPreds_rf <- predict(fit_rf, treatedTest)
# Accuracy for training set
mean(trainingPreds_rf == treatedTrain$Y_AcceptedOffer) # 0.68875

# Accuracy for Test set
mean(testingPreds_rf == treatedTest$Y_AcceptedOffer) # 0.6475
# Get the conf Matrix
confusionMatrix(trainingPreds_rf, treatedTrain$Y_AcceptedOffer)
confusionMatrix(testingPreds_rf, treatedTest$Y_AcceptedOffer)

# 4. Make predictions
prospectPreds_rf <- predict(fit_rf, treatedProspects, type = "prob")

prospectClasses_rf <- ifelse(prospectPreds_rf >= cutoff, 1, 0)
tail(prospectClasses_rf)
# 5. Join probabilities back to ID
prospectsResults_rf <- data.frame(joinProspects$HHuniqueID,
  probablity = prospectPreds_rf
)
head(prospectsResults_rf, 100)

# 6. Identify the top 100 "success" class probabilities from prospectsResults
head(prospectsResults_rf, 100)
prospectsResults_rf_sorted <- prospectsResults_rf %>% arrange(desc(probablity.Accepted))
head(prospectsResults_rf_sorted, 100)


######################################################
# Most Accurate Model GML and this are the TOP TEN
######################################################
# THIS CODE IS PART OF THE MODEL and separated for response
# 4. Make predictions
prospectPreds_glm <- predict(fit_glm, treatedProspects, type = "prob")
prospectPreds_glm
# 5. Join probabilities back to ID
prospectsResults_glm <- data.frame(joinProspects$HHuniqueID,
  probablity = prospectPreds_glm
)
prospectsResults_glm

# 6. Identify the top 100 "success" class probabilities from prospectsResults
head(prospectsResults_glm, 100)
prospectsResults_glm_sorted <- prospectsResults_glm %>% arrange(desc(probablity.Accepted))
top <- head(prospectsResults_glm_sorted, 100)

top
write.csv(top, "topPropspect.csv", row.names = FALSE, quote = FALSE)