library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(ggsci)
library(gridExtra)
library(grid)
library(viridis)
library(MASS)
library(MVN)
library(randomForest)

set.seed(42)

#########################################################################################################################################
# Load and Process Data
#########################################################################################################################################
dataset_sam <- read.csv("EEG_Dataset_SAM.csv")
dataset_mw1 <- read.csv("EEG_Dataset_MW1.csv")
dataset_mw2 <- read.csv("EEG_Dataset_MW2.csv")
dataset_stress <- read.csv("EEG_Dataset_Stress.csv")

dataset_sam$Subject <- NULL
dataset_sam$Index <- NULL
dataset_sam$Task <- NULL
dataset_sam$Trial <- NULL

dataset_mw1$HR <- NULL
dataset_mw1$HRV <- NULL
dataset_mw1$Subject <- NULL
dataset_mw1$Test <- NULL
dataset_mw1$Score <- NULL
dataset_mw1$Index <- NULL

dataset_mw2$HR <- NULL
dataset_mw2$HRV <- NULL
dataset_mw2$Subject <- NULL
dataset_mw2$Test <- NULL
dataset_mw2$Difficulty <- NULL
dataset_mw2$Index <- NULL

dataset_stress$HR <- NULL
dataset_stress$HRV <- NULL
dataset_stress$BPS <- NULL
dataset_stress$BPD <- NULL
dataset_stress$PR <- NULL
dataset_stress$PhysicalDemand <- NULL
dataset_stress$TemporalDemand <- NULL
dataset_stress$MentalDemand <- NULL
dataset_stress$Effort <- NULL
dataset_stress$Frustration <- NULL
dataset_stress$Fatigue <- NULL
dataset_stress$Subject <- NULL
dataset_stress$Index <- NULL

alpha_frontal_sam <- rowMeans(dataset_sam[,c("AlphaFp1", "AlphaFp2", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8", "AlphaFz")])
beta_frontal_sam  <- rowMeans(dataset_sam[,c("BetaFp1", "BetaFp2", "BetaF3", "BetaF4", "BetaF7", "BetaF8", "BetaFz")])
delta_frontal_sam <- rowMeans(dataset_sam[,c("DeltaFp1", "DeltaFp2", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8", "DeltaFz")])
theta_frontal_sam <- rowMeans(dataset_sam[,c("ThetaFp1", "ThetaFp2", "ThetaF3", "ThetaF4", "ThetaF7", "ThetaF8", "ThetaFz")])
gamma_frontal_sam <- rowMeans(dataset_sam[,c("GammaFp1", "GammaFp2", "GammaF3", "GammaF4", "GammaF7", "GammaF8", "GammaFz")])
alpha_central_sam <- rowMeans(dataset_sam[,c("AlphaC3", "AlphaC4", "AlphaCz")])
beta_central_sam <- rowMeans(dataset_sam[,c("BetaC3", "BetaC4", "BetaCz")])
delta_central_sam <- rowMeans(dataset_sam[,c("DeltaC3", "DeltaC4", "DeltaCz")])
theta_central_sam <- rowMeans(dataset_sam[,c("ThetaC3", "ThetaC4", "ThetaCz")])
gamma_central_sam <- rowMeans(dataset_sam[,c("GammaC3", "GammaC4", "GammaCz")])
alpha_parietal_sam <- rowMeans(dataset_sam[,c("AlphaP3", "AlphaP4", "AlphaP7", "AlphaP8", "AlphaPz")])
beta_parietal_sam <- rowMeans(dataset_sam[,c("BetaP3", "BetaP4", "BetaP7", "BetaP8", "BetaPz")])
delta_parietal_sam <- rowMeans(dataset_sam[,c("DeltaP3", "DeltaP4", "DeltaP7", "DeltaP8", "DeltaPz")])
theta_parietal_sam <- rowMeans(dataset_sam[,c("ThetaP3", "ThetaP4", "ThetaP7", "ThetaP8", "ThetaPz")])
gamma_parietal_sam <- rowMeans(dataset_sam[,c("GammaP3", "GammaP4", "GammaP7", "GammaP8", "GammaPz")])
alpha_occipital_sam <- rowMeans(dataset_sam[,c("AlphaO1", "AlphaO2")])
beta_occipital_sam <- rowMeans(dataset_sam[,c("BetaO1", "BetaO2")])
delta_occipital_sam <- rowMeans(dataset_sam[,c("DeltaO1", "DeltaO2")])
theta_occipital_sam <- rowMeans(dataset_sam[,c("ThetaO1", "ThetaO2")])
gamma_occipital_sam <- rowMeans(dataset_sam[,c("GammaO1", "GammaO2")])
alpha_temporal_sam <- rowMeans(dataset_sam[,c("AlphaT7", "AlphaT8")])
beta_temporal_sam <- rowMeans(dataset_sam[,c("BetaT7", "BetaT8")])
delta_temporal_sam <- rowMeans(dataset_sam[,c("DeltaT7", "DeltaT8")])
theta_temporal_sam <- rowMeans(dataset_sam[,c("ThetaT7", "ThetaT8")])
gamma_temporal_sam <- rowMeans(dataset_sam[,c("GammaT7", "GammaT8")])

alpha_frontal_mw1 <- rowMeans(dataset_mw1[,c("AlphaAF3", "AlphaAF4", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8")])
beta_frontal_mw1  <- rowMeans(dataset_mw1[,c("BetaAF3", "BetaAF4", "BetaF3", "BetaF4", "BetaF7", "BetaF8")])
delta_frontal_mw1 <- rowMeans(dataset_mw1[,c("DeltaAF3", "DeltaAF4", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8")])
theta_frontal_mw1 <- rowMeans(dataset_mw1[,c("ThetaAF3", "ThetaAF4", "ThetaF3", "ThetaF4", "ThetaF7", "ThetaF8")])
gamma_frontal_mw1 <- rowMeans(dataset_mw1[,c("GammaAF3", "GammaAF4", "GammaF3", "GammaF4", "GammaF7", "GammaF8")])
alpha_central_mw1 <- rowMeans(dataset_mw1[,c("AlphaFC5", "AlphaFC6")])
beta_central_mw1 <- rowMeans(dataset_mw1[,c("BetaFC5", "BetaFC6")])
delta_central_mw1 <- rowMeans(dataset_mw1[,c("DeltaFC5", "DeltaFC6")])
theta_central_mw1 <- rowMeans(dataset_mw1[,c("ThetaFC5", "ThetaFC6")])
gamma_central_mw1 <- rowMeans(dataset_mw1[,c("GammaFC5", "GammaFC6")])
alpha_parietal_mw1 <- rowMeans(dataset_mw1[,c("AlphaP7", "AlphaP8")])
beta_parietal_mw1 <- rowMeans(dataset_mw1[,c("BetaP7", "BetaP8")])
delta_parietal_mw1 <- rowMeans(dataset_mw1[,c("DeltaP7", "DeltaP8")])
theta_parietal_mw1 <- rowMeans(dataset_mw1[,c("ThetaP7", "ThetaP8")])
gamma_parietal_mw1 <- rowMeans(dataset_mw1[,c("GammaP7", "GammaP8")])
alpha_occipital_mw1 <- rowMeans(dataset_mw1[,c("AlphaO1", "AlphaO2")])
beta_occipital_mw1 <- rowMeans(dataset_mw1[,c("BetaO1", "BetaO2")])
delta_occipital_mw1 <- rowMeans(dataset_mw1[,c("DeltaO1", "DeltaO2")])
theta_occipital_mw1 <- rowMeans(dataset_mw1[,c("ThetaO1", "ThetaO2")])
gamma_occipital_mw1 <- rowMeans(dataset_mw1[,c("GammaO1", "GammaO2")])
alpha_temporal_mw1 <- rowMeans(dataset_mw1[,c("AlphaT7", "AlphaT8")])
beta_temporal_mw1 <- rowMeans(dataset_mw1[,c("BetaT7", "BetaT8")])
delta_temporal_mw1 <- rowMeans(dataset_mw1[,c("DeltaT7", "DeltaT8")])
theta_temporal_mw1 <- rowMeans(dataset_mw1[,c("ThetaT7", "ThetaT8")])
gamma_temporal_mw1 <- rowMeans(dataset_mw1[,c("GammaT7", "GammaT8")])

alpha_frontal_mw2 <- rowMeans(dataset_mw2[,c("AlphaAF3", "AlphaAF4", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8")])
beta_frontal_mw2  <- rowMeans(dataset_mw2[,c("BetaAF3", "BetaAF4", "BetaF3", "BetaF4", "BetaF7", "BetaF8")])
delta_frontal_mw2 <- rowMeans(dataset_mw2[,c("DeltaAF3", "DeltaAF4", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8")])
theta_frontal_mw2 <- rowMeans(dataset_mw2[,c("ThetaAF3", "ThetaAF4", "ThetaF3", "ThetaF4", "ThetaF7", "ThetaF8")])
gamma_frontal_mw2 <- rowMeans(dataset_mw2[,c("GammaAF3", "GammaAF4", "GammaF3", "GammaF4", "GammaF7", "GammaF8")])
alpha_central_mw2 <- rowMeans(dataset_mw2[,c("AlphaFC5", "AlphaFC6")])
beta_central_mw2 <- rowMeans(dataset_mw2[,c("BetaFC5", "BetaFC6")])
delta_central_mw2 <- rowMeans(dataset_mw2[,c("DeltaFC5", "DeltaFC6")])
theta_central_mw2 <- rowMeans(dataset_mw2[,c("ThetaFC5", "ThetaFC6")])
gamma_central_mw2 <- rowMeans(dataset_mw2[,c("GammaFC5", "GammaFC6")])
alpha_parietal_mw2 <- rowMeans(dataset_mw2[,c("AlphaP7", "AlphaP8")])
beta_parietal_mw2 <- rowMeans(dataset_mw2[,c("BetaP7", "BetaP8")])
delta_parietal_mw2 <- rowMeans(dataset_mw2[,c("DeltaP7", "DeltaP8")])
theta_parietal_mw2 <- rowMeans(dataset_mw2[,c("ThetaP7", "ThetaP8")])
gamma_parietal_mw2 <- rowMeans(dataset_mw2[,c("GammaP7", "GammaP8")])
alpha_occipital_mw2 <- rowMeans(dataset_mw2[,c("AlphaO1", "AlphaO2")])
beta_occipital_mw2 <- rowMeans(dataset_mw2[,c("BetaO1", "BetaO2")])
delta_occipital_mw2 <- rowMeans(dataset_mw2[,c("DeltaO1", "DeltaO2")])
theta_occipital_mw2 <- rowMeans(dataset_mw2[,c("ThetaO1", "ThetaO2")])
gamma_occipital_mw2 <- rowMeans(dataset_mw2[,c("GammaO1", "GammaO2")])
alpha_temporal_mw2 <- rowMeans(dataset_mw2[,c("AlphaT7", "AlphaT8")])
beta_temporal_mw2 <- rowMeans(dataset_mw2[,c("BetaT7", "BetaT8")])
delta_temporal_mw2 <- rowMeans(dataset_mw2[,c("DeltaT7", "DeltaT8")])
theta_temporal_mw2 <- rowMeans(dataset_mw2[,c("ThetaT7", "ThetaT8")])
gamma_temporal_mw2 <- rowMeans(dataset_mw2[,c("GammaT7", "GammaT8")])

alpha_frontal_stress <- rowMeans(dataset_stress[,c("AlphaFp1", "AlphaFp2", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8", "AlphaFz")])
beta_frontal_stress  <- rowMeans(dataset_stress[,c("BetaFp1", "BetaFp2", "BetaF3", "BetaF4", "BetaF7", "BetaF8", "BetaFz")])
delta_frontal_stress <- rowMeans(dataset_stress[,c("DeltaFp1", "DeltaFp2", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8", "DeltaFz")])
theta_frontal_stress <- rowMeans(dataset_stress[,c("ThetaFp1", "ThetaFp2", "ThetaF3", "ThetaF4", "ThetaF7", "ThetaF8", "ThetaFz")])
gamma_frontal_stress <- rowMeans(dataset_stress[,c("GammaFp1", "GammaFp2", "GammaF3", "GammaF4", "GammaF7", "GammaF8", "GammaFz")])
alpha_central_stress <- rowMeans(dataset_stress[,c("AlphaC3", "AlphaC4", "AlphaCz")])
beta_central_stress <- rowMeans(dataset_stress[,c("BetaC3", "BetaC4", "BetaCz")])
delta_central_stress <- rowMeans(dataset_stress[,c("DeltaC3", "DeltaC4", "DeltaCz")])
theta_central_stress <- rowMeans(dataset_stress[,c("ThetaC3", "ThetaC4", "ThetaCz")])
gamma_central_stress <- rowMeans(dataset_stress[,c("GammaC3", "GammaC4", "GammaCz")])
alpha_parietal_stress <- rowMeans(dataset_stress[,c("AlphaP3", "AlphaP4", "AlphaP7", "AlphaP8", "AlphaPz")])
beta_parietal_stress <- rowMeans(dataset_stress[,c("BetaP3", "BetaP4", "BetaP7", "BetaP8", "BetaPz")])
delta_parietal_stress <- rowMeans(dataset_stress[,c("DeltaP3", "DeltaP4", "DeltaP7", "DeltaP8", "DeltaPz")])
theta_parietal_stress <- rowMeans(dataset_stress[,c("ThetaP3", "ThetaP4", "ThetaP7", "ThetaP8", "ThetaPz")])
gamma_parietal_stress <- rowMeans(dataset_stress[,c("GammaP3", "GammaP4", "GammaP7", "GammaP8", "GammaPz")])
alpha_occipital_stress <- rowMeans(dataset_stress[,c("AlphaO1", "AlphaO2")])
beta_occipital_stress <- rowMeans(dataset_stress[,c("BetaO1", "BetaO2")])
delta_occipital_stress <- rowMeans(dataset_stress[,c("DeltaO1", "DeltaO2")])
theta_occipital_stress <- rowMeans(dataset_stress[,c("ThetaO1", "ThetaO2")])
gamma_occipital_stress <- rowMeans(dataset_stress[,c("GammaO1", "GammaO2")])
alpha_temporal_stress <- rowMeans(dataset_stress[,c("AlphaT7", "AlphaT8")])
beta_temporal_stress <- rowMeans(dataset_stress[,c("BetaT7", "BetaT8")])
delta_temporal_stress <- rowMeans(dataset_stress[,c("DeltaT7", "DeltaT8")])
theta_temporal_stress <- rowMeans(dataset_stress[,c("ThetaT7", "ThetaT8")])
gamma_temporal_stress <- rowMeans(dataset_stress[,c("GammaT7", "GammaT8")])


temp_sam <- cbind(alpha_frontal_sam, beta_frontal_sam, delta_frontal_sam, theta_frontal_sam, gamma_frontal_sam,
              alpha_central_sam, beta_central_sam, delta_central_sam, theta_central_sam, gamma_central_sam,
              alpha_parietal_sam, beta_parietal_sam, delta_parietal_sam, theta_parietal_sam, gamma_parietal_sam,
              alpha_occipital_sam, beta_occipital_sam, delta_occipital_sam, theta_occipital_sam, gamma_occipital_sam,
              alpha_temporal_sam, beta_temporal_sam, delta_temporal_sam, theta_temporal_sam, gamma_temporal_sam, dataset_sam$Stress)
temp_stress <- cbind(alpha_frontal_stress, beta_frontal_stress, delta_frontal_stress, theta_frontal_stress, gamma_frontal_stress,
                     alpha_central_stress, beta_central_stress, delta_central_stress, theta_central_stress, gamma_central_stress,
                     alpha_parietal_stress, beta_parietal_stress, delta_parietal_stress, theta_parietal_stress, gamma_parietal_stress,
                     alpha_occipital_stress, beta_occipital_stress, delta_occipital_stress, theta_occipital_stress, gamma_occipital_stress,
                     alpha_temporal_stress, beta_temporal_stress, delta_temporal_stress, theta_temporal_stress, gamma_temporal_stress, dataset_stress$Stress)
temp_mw1 <- cbind(alpha_frontal_mw1, beta_frontal_mw1, delta_frontal_mw1, theta_frontal_mw1, gamma_frontal_mw1,
                  alpha_central_mw1, beta_central_mw1, delta_central_mw1, theta_central_mw1, gamma_central_mw1,
                  alpha_parietal_mw1, beta_parietal_mw1, delta_parietal_mw1, theta_parietal_mw1, gamma_parietal_mw1,
                  alpha_occipital_mw1, beta_occipital_mw1, delta_occipital_mw1, theta_occipital_mw1, gamma_occipital_mw1,
                  alpha_temporal_mw1, beta_temporal_mw1, delta_temporal_mw1, theta_temporal_mw1, gamma_temporal_mw1, dataset_mw1$Stress)
temp_mw2 <- cbind(alpha_frontal_mw2, beta_frontal_mw2, delta_frontal_mw2, theta_frontal_mw2, gamma_frontal_mw2,
                  alpha_central_mw2, beta_central_mw2, delta_central_mw2, theta_central_mw2, gamma_central_mw2,
                  alpha_parietal_mw2, beta_parietal_mw2, delta_parietal_mw2, theta_parietal_mw2, gamma_parietal_mw2,
                  alpha_occipital_mw2, beta_occipital_mw2, delta_occipital_mw2, theta_occipital_mw2, gamma_occipital_mw2,
                  alpha_temporal_mw2, beta_temporal_mw2, delta_temporal_mw2, theta_temporal_mw2, gamma_temporal_mw2, dataset_mw2$Stress)

data <- rbind(temp_sam, temp_mw1, temp_stress, temp_mw2)
data <- as.data.frame(data)
names(data) <- c("Alpha Frontal", "Beta Frontal", "Delta Frontal", "Theta Frontal", "Gamma Frontal",
                 "Alpha Central", "Beta Central", "Delta Central", "Theta Central", "Gamma Central",
                 "Alpha Parietal", "Beta Parietal", "Delta Parietal", "Theta Parietal", "Gamma Parietal",
                 "Alpha Occipital", "Beta Occipital", "Delta Occipital", "Theta Occipital", "Gamma Occipital",
                 "Alpha Temporal", "Beta Temporal", "Delta Temporal", "Theta Temporal", "Gamma Temporal", "Stress")
CM = cor(data, method="spearman")
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
synthetic_dataset <- generate_correlated_rows(data, CM, 0.50, required_samples * 2)
synthetic_dataset <- unique(synthetic_dataset)
synthetic_dataset <- synthetic_dataset[1:required_samples, ]

CMN = cor(synthetic_dataset, method="spearman")
ggcorrplot(CMN, lab = TRUE, type = "lower", lab_size = 3) + theme(text = element_text(family = "Times New Roman", face="bold"))

data$ID <- seq.int(nrow(data))
synthetic_dataset$ID <- seq.int(nrow(synthetic_dataset))
write.csv(data, "Original_Dataset_Merged.csv", row.names = FALSE)
write.csv(synthetic_dataset, "Synthetic_Dataset_Merged.csv", row.names = FALSE)

#######################################################################################################################################
# Distribution Analysis
#######################################################################################################################################
original <- read.csv("Original_Dataset_Merged.csv")
synthetic <- read.csv("Synthetic_Dataset_Merged.csv")
o_frontal <- original[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
s_frontal <- synthetic[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
o_frontal$Group <- "Original"
s_frontal$Group <- "Synthetic"
merged_frontal <- rbind(o_frontal, s_frontal)
merged_frontal$Region <- "Frontal"
merged_frontal$Value <- rowMeans(merged_frontal[, 1:5])
o_central <- original[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
s_central <- synthetic[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
o_central$Group <- "Original"
s_central$Group <- "Synthetic"
merged_central <- rbind(o_central, s_central)
merged_central$Region <- "Central"
merged_central$Value <- rowMeans(merged_central[, 1:5])
o_parietal <- original[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
s_parietal <- synthetic[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
o_parietal$Group <- "Original"
s_parietal$Group <- "Synthetic"
merged_parietal <- rbind(o_parietal, s_parietal)
merged_parietal$Region <- "Parietal"
merged_parietal$Value <- rowMeans(merged_parietal[, 1:5])
o_occipital <- original[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
s_occipital <- synthetic[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
o_occipital$Group <- "Original"
s_occipital$Group <- "Synthetic"
merged_occipital <- rbind(o_occipital, s_occipital)
merged_occipital$Region <- "Occipital"
merged_occipital$Value <- rowMeans(merged_occipital[, 1:5])
o_temporal <- original[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
s_temporal <- synthetic[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
o_temporal$Group <- "Original"
s_temporal$Group <- "Synthetic"
merged_temporal <- rbind(o_temporal, s_temporal)
merged_temporal$Region <- "Temporal"
merged_temporal$Value <- rowMeans(merged_temporal[, 1:5])
merged <- rbind(merged_frontal[, c("Group", "Region", "Value")], merged_central[, c("Group", "Region", "Value")], 
                merged_parietal[, c("Group", "Region", "Value")], merged_occipital[, c("Group", "Region", "Value")], 
                merged_temporal[, c("Group", "Region", "Value")])

# remove outliers
Q1 <- quantile(merged$Value, 0.25)
Q3 <- quantile(merged$Value, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
merged <- merged[merged$Value >= lower_bound & merged$Value <= upper_bound, ]

ggplot(merged, aes(Value, fill = Group)) + 
  geom_density(alpha = 0.7) +
  facet_grid(Region ~ .) +
  scale_fill_manual(values = c("Original" = "deepskyblue2", "Synthetic" = "springgreen4")) +
  theme_classic() + 
  labs(x = "Value", y = "Density", title = "Merged Dataset", subtitle =  "(Original vs. Synthetic)") +
  theme(
    axis.title = element_text(size = 16, family="Times New Roman", face="bold"),
    axis.text = element_text(size = 11, family = "Times New Roman", face="bold"),
    plot.title = element_text(family = "Times New Roman", face="bold", size=16, hjust = 0.5),
    plot.subtitle = element_text(family = "Times New Roman", size=10, hjust = 0.5),
    legend.text = element_text(family = "Times New Roman", face="bold", size=12),
    legend.title = element_text(family = "Times New Roman", face="bold", size=14, hjust = 0.5),
    strip.text.y = element_text(family = "Times New Roman", size=12)
  )

#######################################################################################################################################
# Statistical Analysis
#######################################################################################################################################
original <- read.csv("Original_Dataset_Merged.csv")
synthetic <- read.csv("Synthetic_Dataset_Merged.csv")
original$ID <- NULL
synthetic$ID <- NULL

# train a model on original to predict stress
set.seed(42)
rf <- randomForest(Stress ~ ., data=original)

# predict on synthetic
yhat <- predict(rf, synthetic[1:300, 1:25])
rmse <- sqrt(mean((synthetic$Stress - yhat)^2)) # 0.12 - good detection of stress on synthetic

# train a model on original to predict stress
set.seed(42)
rf <- randomForest(Stress ~ ., data=original)

# predict on synthetic
yhat <- predict(rf, synthetic[1:300, 1:25])
rmse <- sqrt(mean((synthetic$Stress - yhat)^2)) # 0.12 - good detection of stress on synthetic

# repeat across
set.seed(42)
rf <- randomForest(Stress ~ ., data=synthetic)
yhat <- predict(rf, original[, 1:25])
rmse <- sqrt(mean((original$Stress - yhat)^2)) # 0.32 


