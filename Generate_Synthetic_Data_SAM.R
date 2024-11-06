library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(ggsci)
library(gridExtra)
library(grid)
library(viridis)
library(MASS)

set.seed(42)

#########################################################################################################################################
# Load and Process Data
#########################################################################################################################################
data <- read.csv("EEG_Dataset_SAM.csv")
temp <- data
temp$Index <- NULL
subjects <- temp$Subject
temp$Subject <- NULL

# frontal
alpha_frontal <- rowMeans(temp[,c("AlphaFp1", "AlphaFp2", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8", "AlphaFz")])
beta_frontal  <- rowMeans(temp[,c("BetaFp1", "BetaFp2", "BetaF3", "BetaF4", "BetaF7", "BetaF8", "BetaFz")])
delta_frontal <- rowMeans(temp[,c("DeltaFp1", "DeltaFp2", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8", "DeltaFz")])
theta_frontal <- rowMeans(temp[,c("ThetaFp1", "ThetaFp2", "ThetaF3", "ThetaF4", "ThetaF7", "ThetaF8", "ThetaFz")])
gamma_frontal <- rowMeans(temp[,c("GammaFp1", "GammaFp2", "GammaF3", "GammaF4", "GammaF7", "GammaF8", "GammaFz")])
# central
alpha_central <- rowMeans(temp[,c("AlphaC3", "AlphaC4", "AlphaCz")])
beta_central <- rowMeans(temp[,c("BetaC3", "BetaC4", "BetaCz")])
delta_central <- rowMeans(temp[,c("DeltaC3", "DeltaC4", "DeltaCz")])
theta_central <- rowMeans(temp[,c("ThetaC3", "ThetaC4", "ThetaCz")])
gamma_central <- rowMeans(temp[,c("GammaC3", "GammaC4", "GammaCz")])
# parietal
alpha_parietal <- rowMeans(temp[,c("AlphaP3", "AlphaP4", "AlphaP7", "AlphaP8", "AlphaPz")])
beta_parietal <- rowMeans(temp[,c("BetaP3", "BetaP4", "BetaP7", "BetaP8", "BetaPz")])
delta_parietal <- rowMeans(temp[,c("DeltaP3", "DeltaP4", "DeltaP7", "DeltaP8", "DeltaPz")])
theta_parietal <- rowMeans(temp[,c("ThetaP3", "ThetaP4", "ThetaP7", "ThetaP8", "ThetaPz")])
gamma_parietal <- rowMeans(temp[,c("GammaP3", "GammaP4", "GammaP7", "GammaP8", "GammaPz")])
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

temp$Relax <- ifelse(temp$Task == "Relax", 1, 0)
temp$Mirror <- ifelse(temp$Task == "Mirror", 1, 0)
temp$Arithmetic <- ifelse(temp$Task == "Arithmetic", 1, 0)
temp$Stroop <- ifelse(temp$Task == "Stroop", 1, 0)
temp$Task <- NULL

temp <- cbind(alpha_frontal, beta_frontal, delta_frontal, theta_frontal, gamma_frontal,
              alpha_central, beta_central, delta_central, theta_central, gamma_central,
              alpha_parietal, beta_parietal, delta_parietal, theta_parietal, gamma_parietal,
              alpha_occipital, beta_occipital, delta_occipital, theta_occipital, gamma_occipital,
              alpha_temporal, beta_temporal, delta_temporal, theta_temporal, gamma_temporal,
              temp$Stress, temp$Relax, temp$Mirror, temp$Arithmetic, temp$Stroop)
temp <- as.data.frame(temp)
names(temp) <- c("Alpha Frontal", "Beta Frontal", "Delta Frontal", "Theta Frontal", "Gamma Frontal",
                 "Alpha Central", "Beta Central", "Delta Central", "Theta Central", "Gamma Central",
                 "Alpha Parietal", "Beta Parietal", "Delta Parietal", "Theta Parietal", "Gamma Parietal",
                 "Alpha Occipital", "Beta Occipital", "Delta Occipital", "Theta Occipital", "Gamma Occipital",
                 "Alpha Temporal", "Beta Temporal", "Delta Temporal", "Theta Temporal", "Gamma Temporal",
                 "Stress", "Relax", "Mirror", "Arithmetic", "Stroop")

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
write.csv(temp, "Original_Dataset_SAM.csv", row.names = FALSE)
write.csv(synthetic_dataset, "Synthetic_Dataset_SAM.csv", row.names = FALSE)
