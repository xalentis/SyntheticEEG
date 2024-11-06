library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(ggsci)
library(gridExtra)
library(grid)
library(viridis)
library(MASS)
library(scales)

set.seed(42)

#########################################################################################################################################
# Load and Process Data
#########################################################################################################################################
data <- read.csv("EEG_Dataset_MW1.csv")
temp <- data
temp$Index <- NULL
subjects <- temp$Subject
temp$Subject <- NULL
temp$Stress <- rescale(temp$Stress, to=c(0,1))

# frontal
alpha_frontal <- rowMeans(temp[,c("AlphaAF3", "AlphaAF4", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8")])
beta_frontal  <- rowMeans(temp[,c("BetaAF3", "BetaAF4", "BetaF3", "BetaF4", "BetaF7", "BetaF8")])
delta_frontal <- rowMeans(temp[,c("DeltaAF3", "DeltaAF4", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8")])
theta_frontal <- rowMeans(temp[,c("ThetaAF3", "ThetaAF4", "ThetaF3", "ThetaF4", "ThetaF7", "ThetaF8")])
gamma_frontal <- rowMeans(temp[,c("GammaAF3", "GammaAF4", "GammaF3", "GammaF4", "GammaF7", "GammaF8")])
# central
alpha_central <- rowMeans(temp[,c("AlphaFC5", "AlphaFC6")])
beta_central <- rowMeans(temp[,c("BetaFC5", "BetaFC6")])
delta_central <- rowMeans(temp[,c("DeltaFC5", "DeltaFC6")])
theta_central <- rowMeans(temp[,c("ThetaFC5", "ThetaFC6")])
gamma_central <- rowMeans(temp[,c("GammaFC5", "GammaFC6")])
# parietal
alpha_parietal <- rowMeans(temp[,c("AlphaP7", "AlphaP8")])
beta_parietal <- rowMeans(temp[,c("BetaP7", "BetaP8")])
delta_parietal <- rowMeans(temp[,c("DeltaP7", "DeltaP8")])
theta_parietal <- rowMeans(temp[,c("ThetaP7", "ThetaP8")])
gamma_parietal <- rowMeans(temp[,c("GammaP7", "GammaP8")])
# occipital
alpha_occipital <- rowMeans(temp[,c("AlphaO1", "AlphaO2")])
beta_occipital <- rowMeans(temp[,c("BetaO1", "BetaO2")])
delta_occipital <- rowMeans(temp[,c("DeltaO1", "DeltaO2")])
theta_occipital <- rowMeans(temp[,c("ThetaO1", "ThetaO2")])
gamma_occipital <- rowMeans(temp[,c("GammaO1", "GammaO2")])
# temporal
alpha_temporal <- rowMeans(temp[,c("AlphaT7", "AlphaT8")])
beta_temporal <- rowMeans(temp[,c("BetaT7", "BetaT8")])
delta_temporal <- rowMeans(temp[,c("DeltaT7", "DeltaT8")])
theta_temporal <- rowMeans(temp[,c("ThetaT7", "ThetaT8")])
gamma_temporal <- rowMeans(temp[,c("GammaT7", "GammaT8")])

temp <- cbind(alpha_frontal, beta_frontal, delta_frontal, theta_frontal, gamma_frontal,
              alpha_central, beta_central, delta_central, theta_central, gamma_central,
              alpha_parietal, beta_parietal, delta_parietal, theta_parietal, gamma_parietal,
              alpha_occipital, beta_occipital, delta_occipital, theta_occipital, gamma_occipital,
              alpha_temporal, beta_temporal, delta_temporal, theta_temporal, gamma_temporal, 
              temp$Score, temp$HR, temp$HRV, temp$Test, temp$Stress)
temp <- as.data.frame(temp)
names(temp)[26] <- "Score"
names(temp)[27] <- "HR"
names(temp)[28] <- "HRV"
names(temp)[29] <- "Test"
names(temp)[30] <- "Stress"
temp$Low <- 0
temp$Medium <- 0
temp$High <- 0
temp[temp$Test == 1,"Low"] <- 1
temp[temp$Test == 2,"Medium"] <- 1
temp[temp$Test == 3,"High"] <- 1
temp$Test <- NULL
names(temp) <- c("Alpha Frontal", "Beta Frontal", "Delta Frontal", "Theta Frontal", "Gamma Frontal",
                 "Alpha Central", "Beta Central", "Delta Central", "Theta Central", "Gamma Central",
                 "Alpha Parietal", "Beta Parietal", "Delta Parietal", "Theta Parietal", "Gamma Parietal",
                 "Alpha Occipital", "Beta Occipital", "Delta Occipital", "Theta Occipital", "Gamma Occipital",
                 "Alpha Temporal", "Beta Temporal", "Delta Temporal", "Theta Temporal", "Gamma Temporal",
                 "Score", "HR", "HRV","Stress", "Low", "Medium", "High")

CM = cor(temp, method="spearman")
ggcorrplot(CM, lab = TRUE, type = "lower", lab_size = 3) + theme(text = element_text(family = "Times New Roman", face="bold"))

#########################################################################################################################################
# Synthetic data generation through random sampling and correlation analysis
#########################################################################################################################################
generate_correlated_rows <- function(DF, CM, T, n_rows) {
  is_within_threshold <- function(sampled_data, CM, T) {
    sampled_corr <- cor(sampled_data, method="spearman")
    return(all(abs(sampled_corr - CM) <= T))
  }
  new_rows <- matrix(NA, nrow = 0, ncol = ncol(DF))
  colnames(new_rows) <- colnames(DF)
  while (nrow(new_rows) < n_rows) {
    set.seed(42 + nrow(new_rows))
    sampled_rows <- DF[sample(1:nrow(DF), n_rows, replace = TRUE), ]
    if (is_within_threshold(sampled_rows, CM, T)) {
      new_rows <- rbind(new_rows, sampled_rows)
    }
  }
  return(as.data.frame(new_rows))
}

required_samples <- 300
synthetic_dataset <- NULL
synthetic_dataset <- generate_correlated_rows(temp, CM, 0.20, required_samples * 2)
synthetic_dataset <- unique(synthetic_dataset)
synthetic_dataset <- synthetic_dataset[1:required_samples, ]

CMN = cor(synthetic_dataset, method="spearman")
ggcorrplot(CMN, lab = TRUE, type = "lower", lab_size = 3) + theme(text = element_text(family = "Times New Roman", face="bold"))

temp$ID <- seq.int(nrow(temp))
synthetic_dataset$ID <- seq.int(nrow(synthetic_dataset))
write.csv(temp, "Original_Dataset_MW1.csv", row.names = FALSE)
write.csv(synthetic_dataset, "Synthetic_Dataset_MW1.csv", row.names = FALSE)
