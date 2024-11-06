library(MVN)
library(vegan)
library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(ggsci)
library(gridExtra)
library(grid)
library(viridis)
library(randomForest)

options(scipen=999)

original <- read.csv("Original_Dataset_SAM.csv")
synthetic <- read.csv("Synthetic_Dataset_SAM.csv")

original$ID <- NULL
synthetic$ID <- NULL
merged <- rbind(original, synthetic)

# remove any linearly dependent columns
qr_decomp <- qr(merged)
linearly_independent <- merged[, qr_decomp$pivot[1:qr_decomp$rank]]

# test for multivariate normality
# based on the Shapiro-Wilk test and is used for smaller sample sizes
royston_test <- mvn(data = linearly_independent, mvnTest = "royston")
print(royston_test$multivariateNormality)
# p<0.000000 so we need to use a non-parametric test like PERMANOVA
# since the assumption of multivariate normality is violated

# Perform PERMANOVA
group <- factor(c(rep('A', nrow(original)), rep('B', nrow(synthetic))))
permanova_result <- adonis2(linearly_independent ~ group, method = "euclidean")
print(permanova_result) # p=0.598, no statistical difference between original and synthetic datasets

# can ML model distinguish between original and synthetic? NO
linearly_independent$Group <- factor(c(rep(0, nrow(original)), rep(1, nrow(synthetic))))
set.seed(42)
rf <- randomForest(Group ~ ., data=linearly_independent) # OOB estimate of  error rate: 47.62%
linearly_independent <- linearly_independent[, 2:23]
set.seed(42)
rf <- randomForest(Group ~ ., data=linearly_independent) # OOB estimate of  error rate: 47.62% - same




