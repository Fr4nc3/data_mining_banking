######################################################
# Francia F. Riesco
# Banking  EDA
######################################################

######################################################
# Libraries that we will use on this process
######################################################
library(dplyr)
library(vtreat)
library(caret)
library(dplyr)
library(rpart.plot) # visualizing
library(MLmetrics)
library(ROSE)
library(ggplot2)
library(ggthemes)
library(randomForest)
options(scipen = 999)
setwd("data_mining_banking/II_National_City_Bank/training")


######################################################
# Load the dataset
######################################################
# CurrentCustomerMktgResults.csv	householdCreditData.csv
# householdAxiomData.csv		householdVehicleData.csv
currentData <- read.csv("CurrentCustomerMktgResults.csv")
newDataSource <- read.csv("householdVehicleData.csv")
newDataCreditSource <- read.csv("householdCreditData.csv")

newDataAxiomSource <- read.csv("householdAxiomData.csv")

######################################################
# Perform a join, need to add other data sets
######################################################
joinData <- left_join(currentData, newDataSource, by = c("HHuniqueID"))
joinData <- left_join(joinData, newDataCreditSource, by = c("HHuniqueID"))
joinData <- left_join(joinData, newDataAxiomSource, by = c("HHuniqueID"))

colnames(joinData)

######################################################
# Check the get summary of the dataset
######################################################

summary(joinData)
table(is.na(joinData$dataID))
table(is.na(joinData$HHuniqueID))
table(is.na(joinData$Communication))
table(is.na(joinData$LastContactDay))
table(is.na(joinData$LastContactMonth))
table(is.na(joinData$NoOfContacts))
table(is.na(joinData$DaysPassed))
table(is.na(joinData$PrevAttempts))
table(is.na(joinData$past_Outcome))
table(is.na(joinData$CallStart))
table(is.na(joinData$CallEnd))
table(is.na(joinData$Y_AcceptedOffer))
table(is.na(joinData$carMake))
table(is.na(joinData$carModel))
table(is.na(joinData$carYr))
table(is.na(joinData$DefaultOnRecord))
table(is.na(joinData$RecentBalance))
table(is.na(joinData$HHInsurance))
table(is.na(joinData$CarLoan))
table(is.na(joinData$headOfhouseholdGender))
table(is.na(joinData$annualDonations))
table(is.na(joinData$EstRace))
table(is.na(joinData$PetsPurchases))
table(is.na(joinData$DigitalHabits_5_AlwaysOn))
table(is.na(joinData$AffluencePurchases))
table(is.na(joinData$Age))
table(is.na(joinData$Job))
table(is.na(joinData$Marital))
table(is.na(joinData$Education))

######################################################
# check  empty values
######################################################

table(joinData$dataID == "")
table(joinData$HHuniqueID == "")
table(joinData$Communication == "")
table(joinData$LastContactDay == "")
table(joinData$LastContactMonth == "")
table(joinData$NoOfContacts == "")
table(joinData$DaysPassed == "")
table(joinData$PrevAttempts == "")
table(joinData$past_Outcome == "")
table(joinData$CallStart == "")
table(joinData$CallEnd == "")
table(joinData$Y_AcceptedOffer == "")
table(joinData$carMake == "")
table(joinData$carModel == "")
table(joinData$carYr == "")
table(joinData$DefaultOnRecord == "")
table(joinData$RecentBalance == "")
table(joinData$HHInsurance == "")
table(joinData$CarLoan == "")
table(joinData$headOfhouseholdGender == "")
table(joinData$annualDonations == "")
table(joinData$EstRace == "")
table(joinData$PetsPurchases == "")
table(joinData$DigitalHabits_5_AlwaysOn == "")
table(joinData$AffluencePurchases == "")
table(joinData$Age == "")
table(joinData$Job == "")
table(joinData$Marital == "")
table(joinData$Education == "")

unique(joinData$Communication)
unique(joinData$LastContactDay)
unique(joinData$LastContactMonth)
unique(joinData$NoOfContacts)
unique(joinData$DaysPassed)
unique(joinData$PrevAttempts)
unique(joinData$past_Outcome)
unique(joinData$Y_AcceptedOffer)
unique(joinData$carMake)
unique(joinData$carModel)
unique(joinData$carYr)
unique(joinData$DefaultOnRecord)
unique(joinData$RecentBalance)
unique(joinData$HHInsurance)
unique(joinData$CarLoan)
unique(joinData$headOfhouseholdGender)
unique(joinData$annualDonations)
unique(joinData$EstRace)
unique(joinData$PetsPurchases)
unique(joinData$DigitalHabits_5_AlwaysOn)
unique(joinData$AffluencePurchases)
unique(joinData$Age)
unique(joinData$Job)
unique(joinData$Marital)
unique(joinData$Education)

######################################################
# Fields with NA ENTRIES
######################################################
table(is.na(joinData$Communication))
table(is.na(joinData$past_Outcome))
table(is.na(joinData$carYr))
table(is.na(joinData$Job))
table(is.na(joinData$Education))


######################################################
# Fields with empty values
######################################################

table(joinData$carMake == "")
table(joinData$carModel == "")
table(joinData$EstRace == "")
table(joinData$annualDonations == "")


######################################################
# Analyzing the Categorical values in the Missing value fields.
######################################################

# categorial values missing will be update with the mode of the field
# numerical values will be update with the mean of the field
# create mode function
Mode <- function(x) {
        ux <- na.omit(unique(x))
        tab <- tabulate(match(x, ux))
        ux[tab == max(tab)]
}
# reference https://www.codingprof.com/how-to-replace-nas-with-the-mode-most-frequent-value-in-r/

joinData$Communication[is.na(joinData$Communication)] <- Mode(joinData$Communication)
joinData$past_Outcome[is.na(joinData$past_Outcome)] <- "unknown" # missing too many touse mode
joinData$carYr[is.na(joinData$carYr)] <- mean(joinData$carYr, na.rm = TRUE)
joinData$Job[is.na(joinData$Job)] <- Mode(joinData$Job)
joinData$Education[is.na(joinData$Education)] <- Mode(joinData$Education)

# we don't add model on this fields
joinData$carMake[joinData$carMake == ""] <- "unknown"
joinData$carModel[joinData$carModel == ""] <- "unknown"

# remove these columns which have so many missing
joinData <- subset(joinData, select = -c(EstRace, annualDonations))


joinData$CallStart <- as.POSIXct(joinData$CallStart,
        format = "%H:%M:%S"
)
joinData$CallEnd <- as.POSIXct(joinData$CallEnd,
        format = "%H:%M:%S"
)
joinData$CallMin <- difftime(joinData$CallEnd, joinData$CallStart, units = "mins")
head(joinData$CallMin)

# use as reference for the presentation
write.csv(joinData, "joinData.csv", row.names = FALSE, quote = FALSE)
names <- names(joinData)
names
######################################################
# Visualization the data
######################################################


barplot(sort(table(joinData$Communication), decreasing = TRUE), main = "Communication")
barplot(sort(table(joinData$LastContactDay), decreasing = TRUE), main = "LastContactDay")
barplot(sort(table(joinData$LastContactMonth), decreasing = TRUE), main = "LastContactMont")
barplot(sort(table(joinData$NoOfContacts), decreasing = TRUE), main = "NoOfContacts")
barplot(sort(table(joinData$DaysPassed), decreasing = TRUE), main = "DaysPassed")
barplot(sort(table(joinData$PrevAttempts), decreasing = TRUE), main = "PrevAttempts")
barplot(sort(table(joinData$past_Outcome), decreasing = TRUE), main = "past_Outcome")
barplot(sort(table(joinData$Y_AcceptedOffer), decreasing = TRUE), main = "Y_AcceptedOffer")
barplot(sort(table(joinData$carMake), decreasing = TRUE), main = "carMake")
barplot(sort(table(joinData$carModel), decreasing = TRUE), main = "carModel")
barplot(sort(table(joinData$carYr), decreasing = TRUE), main = "carYr")
barplot(sort(table(joinData$DefaultOnRecord), decreasing = TRUE), main = "DefaultOnRecord")
barplot(sort(table(joinData$RecentBalance), decreasing = TRUE), main = "RecentBalance")
barplot(sort(table(joinData$HHInsurance), decreasing = TRUE), main = "HHInsurance")
barplot(sort(table(joinData$CarLoan), decreasing = TRUE), main = "CarLoan")
barplot(sort(table(joinData$headOfhouseholdGender), decreasing = TRUE), main = "headOfhouseholdGender")

barplot(sort(table(joinData$PetsPurchases), decreasing = TRUE), main = "PetsPurchases")
barplot(sort(table(joinData$DigitalHabits_5_AlwaysOn), decreasing = TRUE), main = "DigitalHabits_5_AlwaysOn")
barplot(sort(table(joinData$AffluencePurchases), decreasing = TRUE), main = "AffluencePurchases")
barplot(table(joinData$Age), main = "Age")
barplot(sort(table(joinData$Job), decreasing = TRUE), main = "Job")
barplot(sort(table(joinData$Marital), decreasing = TRUE), main = "Marital")
barplot(sort(table(joinData$Education), decreasing = TRUE), main = "Education")


######################################################
# barplot
######################################################
multi.table <- table(joinData$Y_AcceptedOffer, joinData$Age)
multi.table
barplot(multi.table,
        main = "Age Accept Offer", legend.text = TRUE,
        ylab = "Y_AcceptedOffer", xlab = "Age ", col = c("lightgreen", "pink", "lightblue")
)

multi.table <- table(joinData$Y_AcceptedOffer, joinData$Marital)
multi.table
barplot(multi.table,
        main = "Marital Accept Offer", legend.text = TRUE,
        ylab = "Y_AcceptedOffer", xlab = "Marital ", col = c("lightgreen", "pink", "lightblue")
)

multi.table <- table(joinData$Y_AcceptedOffer, joinData$PetsPurchases)
multi.table
barplot(multi.table,
        main = "PetsPurchases Accept Offer", legend.text = TRUE,
        ylab = "Y_AcceptedOffer", xlab = "PetsPurchases ", col = c("lightgreen", "pink", "lightblue")
)

multi.table <- table(joinData$Y_AcceptedOffer, joinData$Job)
multi.table
barplot(multi.table,
        main = "Job Accept Offer", legend.text = TRUE,
        ylab = "Y_AcceptedOffer", xlab = "Job ", col = c("lightgreen", "pink", "lightblue")
)



multi.table <- table(joinData$Y_AcceptedOffer, joinData$Communication)
multi.table
barplot(multi.table,
        main = "Communication Accept Offer", legend.text = TRUE,
        ylab = "Y_AcceptedOffer", xlab = "Communication ", col = c("lightgreen", "pink", "lightblue")
)



multi.table <- table(joinData$Y_AcceptedOffer, joinData$headOfhouseholdGender)
multi.table
barplot(multi.table,
        main = "headOfhouseholdGender Accept Offer", legend.text = TRUE,
        ylab = "Y_AcceptedOffer", xlab = "headOfhouseholdGender ", col = c("lightgreen", "pink", "lightblue")
)



multi.table <- table(joinData$Y_AcceptedOffer, joinData$Education)
multi.table
barplot(multi.table,
        main = "Education Accept Offer", legend.text = TRUE,
        ylab = "Y_AcceptedOffer", xlab = "Education ", col = c("lightgreen", "pink", "lightblue")
)

######################################################
# histograms
######################################################
histogram(~ Y_AcceptedOffer | headOfhouseholdGender, width = 1, xlab = "headOfhouseholdGender ", data = joinData)
histogram(~ Y_AcceptedOffer | Education, width = 1, xlab = " Education", data = joinData)
histogram(~ Y_AcceptedOffer | Communication, width = 1, xlab = "Communication", data = joinData)
histogram(~ Y_AcceptedOffer | Age, width = 1, xlab = "Age", data = joinData)


######################################################
## NOW TO GET PROSPECTIVE CUSTOMER RESULTS
######################################################
# 1. Load Raw Data
prospects <- read.csv("../ProspectiveCustomers.csv")

# 2. Join with external data

joinProspects <- left_join(prospects, newDataSource, by = c("HHuniqueID"))
joinProspects <- left_join(joinProspects, newDataCreditSource, by = c("HHuniqueID"))
joinProspects <- left_join(joinProspects, newDataAxiomSource, by = c("HHuniqueID"))

######################################################
# Check the get summary of the dataset
######################################################

summary(joinProspects)
table(is.na(joinProspects$dataID))
table(is.na(joinProspects$HHuniqueID))
table(is.na(joinProspects$Communication))
table(is.na(joinProspects$LastContactDay))
table(is.na(joinProspects$LastContactMonth))
table(is.na(joinProspects$NoOfContacts))
table(is.na(joinProspects$DaysPassed))
table(is.na(joinProspects$PrevAttempts))
table(is.na(joinProspects$past_Outcome))
table(is.na(joinProspects$CallStart))
table(is.na(joinProspects$CallEnd))
table(is.na(joinProspects$Y_AcceptedOffer))
table(is.na(joinProspects$carMake))
table(is.na(joinProspects$carModel))
table(is.na(joinProspects$carYr))
table(is.na(joinProspects$DefaultOnRecord))
table(is.na(joinProspects$RecentBalance))
table(is.na(joinProspects$HHInsurance))
table(is.na(joinProspects$CarLoan))
table(is.na(joinProspects$headOfhouseholdGender))
table(is.na(joinProspects$annualDonations))
table(is.na(joinProspects$EstRace))
table(is.na(joinProspects$PetsPurchases))
table(is.na(joinProspects$DigitalHabits_5_AlwaysOn))
table(is.na(joinProspects$AffluencePurchases))
table(is.na(joinProspects$Age))
table(is.na(joinProspects$Job))
table(is.na(joinProspects$Marital))
table(is.na(joinProspects$Education))

# remove these columns which have so many missing
joinProspects <- subset(joinProspects, select = -c(EstRace, annualDonations))


# this variable is missing I created with zeroes
joinProspects$CallStart <- as.POSIXct("12:45:22",
        format = "%H:%M:%S"
)
joinProspects$CallEnd <- as.POSIXct("12:45:22",
        format = "%H:%M:%S"
)
joinProspects$CallMin <- difftime(joinProspects$CallEnd, joinProspects$CallStart, units = "mins")
head(joinProspects$CallMin)
######################################################
# NOTE
######################################################
# I wont replace missing data in the prospective to keep it as the original as posible
# use as reference for the presentation
write.csv(joinProspects, "joinProspects.csv", row.names = FALSE, quote = FALSE)