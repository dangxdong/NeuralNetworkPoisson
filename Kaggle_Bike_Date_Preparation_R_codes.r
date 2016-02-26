# 2016-02-26
# This is the note of how the data are transformed and thus prepared for further analysis.
# Record exactly how I process the train.csv. As I should do the same thing with the test.csv.
bike = read.csv("train.csv")
# Extract year, month and hour as new columns
bike$year=as.integer(substr(as.character(bike$datetime), 1,4))
bike$month=as.integer(substr(as.character(bike$datetime), 6,7))
bike$hour=as.integer(substr(as.character(bike$datetime), 12,13))

# install.packages("lubridate")
library(lubridate)

bike$wkday=wday(bike$datetime, label=TRUE)
# check if any pattern:
boxplot(bike$count ~ bike$wkday)

# To reorder the week days from Monday to Sunday:
boxplot(bike$count ~ factor(bike$wkday, levels=levels(bike$wkday)[c(2:7,1)]))



# no strong pattern but could be useful.

# Remove the datetime column
bike$datetime=NULL

# To ease the creation of new columns for categorical features
# Prepare with two dummy columns
bike$id=1:length(bike[,1])
bike$value=1

library(reshape)

# Get the transformed columns of the seasons
seasonMX=data.frame(cast(bike, id ~ season, mean, fill=0))
names(seasonMX) = c("id", "spring", "summer", "fall", "winter")

# Only three seasons are needed as the 4th is non-free
bike=data.frame(bike, seasonMX[,2:4])
bike$season=NULL

# similarly, create the year column:
yearMX=data.frame(cast(bike, id ~ year, mean, fill=0))
names(yearMX) = c("id", "year2011", "year2012")
bike=data.frame(bike, year2011=yearMX[,2])
bike$year=NULL

# similarly, create the weekdays columns
wkdayMX=data.frame(cast(bike, id ~ wkday, mean, fill=0))
bike=data.frame(bike, wkdayMX[,3:8]) # only need six columns!

# Other considerations are wheter to transform "weather" and "month" 
# to categorical columns as above.
# I think the weather is a quasi-numeric feature.
# check by plotting:
boxplot(bike$count ~ bike$weather)
# It's a clear negative correlation, near linear.

# And for month, there would be too many columns in the data set
# check the relation by plotting:
boxplot(bike$count ~ bike$month)
# it's a smooth, slightly non-linear relationship.
# I believe neural network can cope with it as a quasi-numeric feature.

# so I decided not to transform the "weather" and "month" features.

# Removed the dummy columns
bike$value=NULL
bike$id=NULL

# The atemp column (feeled temperature) is highly correlated to the temp column
# A proper way is to created a new column which is the ratio between them.
bike$atempr = bike$atemp / bike$temp

# And then remove atemp
bike$atemp=NULL

# rearrange the column order:
bike = bike[, c(1:4, 23, 5, 6, 10:22, 7, 8, 9)]
# remove the wkday column, because we now have the vectorized ones
bike$wkday=NULL

# So we have the 22-column data frame ready.

# subset the predictors and the outcomes
X = bike[,1:19]
Y = bike[,20:22]

# Split Y into three vectors
Ycasual=Y[,1]
Yregi=Y[,2]
Ytotal=Y[,3]

# Check distribution of the outcomes
hist(bike$casual)
hist(bike$registered)
hist(bike$count)
# so the outcome is count numbers, and quite amenable to Poisson distribution.

# check value scale and variation of the features
colMeans(bike)
lapply(bike, sd)

# There seems no need to do PCA or normalization 

# Stratified splitting into training and validation sets

library(caTools)

set.seed(3030)
split = sample.split(Ytotal, SplitRatio = 0.6)

# split the X data frame
Xtrain = subset(X, split == T)
Xval = subset(X, split == F)

# split the three Y vectors
Ycasualtrain = subset(Ycasual, split == T)
Yregitrain = subset(Yregi, split == T)
Ytotaltrain = subset(Ytotal, split == T)

Ycasualval = subset(Ycasual, split == F)
Yregival = subset(Yregi, split == F)
Ytotalval = subset(Ytotal, split == F)

# split the whole cleaned bike data frame
biketrain=subset(bike, split == T)
bikeval=subset(bike, split == F)

# assemble three pairs (training - valiation) of data frames 
# for use by R packages when needed.
XYtrain_casual = data.frame(Xtrain, casual=Ycasualtrain)
XYtrain_regi = data.frame(Xtrain, registered=Yregitrain)
XYtrain_total = data.frame(Xtrain, total_count=Ytotaltrain)

XYval_casual = data.frame(Xval, casual=Ycasualval)
XYval_regi = data.frame(Xval, registered=Yregival)
XYval_total = data.frame(Xval, total_count=Ytotalval)


# Write all the prepared data frames into csv files:

write.csv(XYtrain_casual, file="XYtrain_casual.csv", row.names=F)
write.csv(XYtrain_regi, file="XYtrain_regi.csv", row.names=F)
write.csv(XYtrain_total, file="XYtrain_total.csv", row.names=F)
write.csv(XYval_casual, file="XYval_casual.csv", row.names=F)
write.csv(XYval_regi, file="XYval_regi.csv", row.names=F)
write.csv(XYval_total, file="XYval_total.csv", row.names=F)

write.csv(Ycasualtrain, file="Ycasualtrain.csv", row.names=F)
write.csv(Yregitrain, file="Yregitrain.csv", row.names=F)
write.csv(Ytotaltrain, file="Ytotaltrain.csv", row.names=F)
write.csv(Ycasualval, file="Ycasualval.csv", row.names=F)
write.csv(Yregival, file="Yregival.csv", row.names=F)
write.csv(Ytotalval, file="Ytotalval.csv", row.names=F)

write.csv(biketrain, file="biketrain.csv", row.names=F)
write.csv(bikeval, file="bikeval.csv", row.names=F)
write.csv(Xtrain, file="Xtrain.csv", row.names=F)
write.csv(Xval, file="Xval.csv", row.names=F)
write.csv(X, file="X.csv", row.names=F)
write.csv(Y, file="Y.csv", row.names=F)

write.csv(Ycasual, file="Ycasual.csv", row.names=F)
write.csv(Yregi, file="Yregi.csv", row.names=F)
write.csv(Ytotal, file="Ytotal.csv", row.names=F)


# The saved csv files will be used in Octave for neural network modelling.


# Below are some trials with R packages.
# For count numbers, it's straightforward to use glm with family=poisson()
# 
# If it's not integer, but continuous, then use Gamma distribtion.
#

# just use XYtrain_total, with total_count as the label.

lmtotal = glm(total_count ~ ., data = XYtrain_total, family="poisson")

summary(lmtotal)

predtrain = exp(predict(lmtotal))
predval= exp(predict(lmtotal, newdata = XYval_total))

# plot and save into png files
png(file = "glm_prediction.png", width=1000, height=600)
par(mfrow=c(1,2))
plot(predtrain, XYtrain_total$total_count, 
     main="Prediction on training data",
     xlab="Predicted counts", ylab="Actual Counts")

plot(predval, XYval_total$total_count, 
     main="Prediction on validation data",
     xlab="Predicted counts", ylab="Actual Counts")
dev.off()

# calculate the RMSLE (Root Mean Squared Log Errors)

calcRMSLE = function (pred, y) {
    m = length(pred)
    predlg = log(pred + 1)
    ylg = log(y + 1)
    cost = 1 / m * sum((predlg-ylg)*(predlg-ylg))
    cost = sum(cost)
    sqrt(cost)
}

RMSLE_train = calcRMSLE(predtrain, XYtrain_total$total_count)
# 1.1961

RMSLE_val = calcRMSLE(predval, XYval_total$total_count)
# 1.2044

# For reference, the best score in RMSLE is 0.32 (Smallest is best).
# The neural network in Octave can reach 0.45 ~ 0.50.

# Also prepare the test.csv for final prediction:

biketest = read.csv("test.csv")

biketest$year=as.integer(substr(as.character(biketest$datetime), 1,4))
biketest$month=as.integer(substr(as.character(biketest$datetime), 6,7))
biketest$hour=as.integer(substr(as.character(biketest$datetime), 12,13))
library(lubridate)
biketest$wkday=wday(biketest$datetime, label=TRUE)
biketest$datetime=NULL
biketest$id=1:length(biketest[,1])
biketest$value=1
library(reshape)
seasonMX=data.frame(cast(biketest, id ~ season, mean, fill=0))
names(seasonMX) = c("id", "spring", "summer", "fall", "winter")
biketest=data.frame(biketest, seasonMX[,2:4])
biketest$season=NULL
yearMX=data.frame(cast(biketest, id ~ year, mean, fill=0))
names(yearMX) = c("id", "year2011", "year2012")
biketest=data.frame(biketest, year2011=yearMX[,2])
biketest$year=NULL
wkdayMX=data.frame(cast(biketest, id ~ wkday, mean, fill=0))
biketest=data.frame(biketest, wkdayMX[,3:8])
biketest$value=NULL
biketest$id=NULL
biketest$atempr = biketest$atemp / biketest$temp
biketest$atemp=NULL
biketest$wkday=NULL
biketest=biketest[,c(1:4, 19, 5:18)]

# write it in csv:
write.csv(biketest, file="test_prepared.csv", row.names=F)

# Later in Octave, read in the file test_prepared.csv as the final X_test matrix.
