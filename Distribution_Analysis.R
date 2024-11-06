library(ggplot2)
library(ggsci)
library(gridExtra)
library(grid)
library(viridis)

options(scipen=999)

#######################################################################################################################################
# EEGStress
#######################################################################################################################################
original_stress <- read.csv("Original_Dataset_Stress.csv")
synthetic_stress <- read.csv("Synthetic_Dataset_Stress.csv")
o_frontal <- original_stress[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
s_frontal <- synthetic_stress[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
o_frontal$Group <- "Original"
s_frontal$Group <- "Synthetic"
merged_frontal <- rbind(o_frontal, s_frontal)
merged_frontal$Region <- "Frontal"
merged_frontal$Value <- rowMeans(merged_frontal[, 1:5])
o_central <- original_stress[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
s_central <- synthetic_stress[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
o_central$Group <- "Original"
s_central$Group <- "Synthetic"
merged_central <- rbind(o_central, s_central)
merged_central$Region <- "Central"
merged_central$Value <- rowMeans(merged_central[, 1:5])
o_parietal <- original_stress[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
s_parietal <- synthetic_stress[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
o_parietal$Group <- "Original"
s_parietal$Group <- "Synthetic"
merged_parietal <- rbind(o_parietal, s_parietal)
merged_parietal$Region <- "Parietal"
merged_parietal$Value <- rowMeans(merged_parietal[, 1:5])
o_occipital <- original_stress[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
s_occipital <- synthetic_stress[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
o_occipital$Group <- "Original"
s_occipital$Group <- "Synthetic"
merged_occipital <- rbind(o_occipital, s_occipital)
merged_occipital$Region <- "Occipital"
merged_occipital$Value <- rowMeans(merged_occipital[, 1:5])
merged <- rbind(merged_frontal[, c("Group", "Region", "Value")], merged_central[, c("Group", "Region", "Value")], 
                merged_parietal[, c("Group", "Region", "Value")], merged_occipital[, c("Group", "Region", "Value")])

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
  labs(x = "Value", y = "Density", title = "Stress Dataset", subtitle =  "(Original vs. Synthetic)") +
  theme(
    axis.title = element_text(size = 16, family="Times New Roman", face="bold"),
    axis.text = element_text(size = 11, family = "Times New Roman", face="bold"),
    plot.title = element_text(family = "Times New Roman", face="bold", size=16, hjust = 0.5),
    plot.subtitle = element_text(family = "Times New Roman", size=10, hjust = 0.5),
    legend.text = element_text(family = "Times New Roman", face="bold", size=12),
    legend.title = element_text(family = "Times New Roman", face="bold", size=14, hjust = 0.5),
    strip.text.y = element_text(family = "Times New Roman", size=12)
  )
merged_stress <- merged

#######################################################################################################################################
# SAM
#######################################################################################################################################
original_sam <- read.csv("Original_Dataset_SAM.csv")
synthetic_sam <- read.csv("Synthetic_Dataset_SAM.csv")
o_frontal <- original_sam[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
s_frontal <- synthetic_sam[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
o_frontal$Group <- "Original"
s_frontal$Group <- "Synthetic"
merged_frontal <- rbind(o_frontal, s_frontal)
merged_frontal$Region <- "Frontal"
merged_frontal$Value <- rowMeans(merged_frontal[, 1:5])
o_central <- original_sam[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
s_central <- synthetic_sam[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
o_central$Group <- "Original"
s_central$Group <- "Synthetic"
merged_central <- rbind(o_central, s_central)
merged_central$Region <- "Central"
merged_central$Value <- rowMeans(merged_central[, 1:5])
o_parietal <- original_sam[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
s_parietal <- synthetic_sam[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
o_parietal$Group <- "Original"
s_parietal$Group <- "Synthetic"
merged_parietal <- rbind(o_parietal, s_parietal)
merged_parietal$Region <- "Parietal"
merged_parietal$Value <- rowMeans(merged_parietal[, 1:5])
o_occipital <- original_sam[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
s_occipital <- synthetic_sam[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
o_occipital$Group <- "Original"
s_occipital$Group <- "Synthetic"
merged_occipital <- rbind(o_occipital, s_occipital)
merged_occipital$Region <- "Occipital"
merged_occipital$Value <- rowMeans(merged_occipital[, 1:5])
o_temporal <- original_sam[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
s_temporal <- synthetic_sam[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
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
  labs(x = "Value", y = "Density", title = "SAM Dataset", subtitle =  "(Original vs. Synthetic)") +
  theme(
    axis.title = element_text(size = 16, family="Times New Roman", face="bold"),
    axis.text = element_text(size = 11, family = "Times New Roman", face="bold"),
    plot.title = element_text(family = "Times New Roman", face="bold", size=16, hjust = 0.5),
    plot.subtitle = element_text(family = "Times New Roman", size=10, hjust = 0.5),
    legend.text = element_text(family = "Times New Roman", face="bold", size=12),
    legend.title = element_text(family = "Times New Roman", face="bold", size=14, hjust = 0.5),
    strip.text.y = element_text(family = "Times New Roman", size=12)
  )
merged_sam <- merged

#######################################################################################################################################
# Mental Workload 1
#######################################################################################################################################
original_mw1 <- read.csv("Original_Dataset_MW1.csv")
synthetic_mw1 <- read.csv("Synthetic_Dataset_MW1.csv")
o_frontal <- original_mw1[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
s_frontal <- synthetic_mw1[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
o_frontal$Group <- "Original"
s_frontal$Group <- "Synthetic"
merged_frontal <- rbind(o_frontal, s_frontal)
merged_frontal$Region <- "Frontal"
merged_frontal$Value <- rowMeans(merged_frontal[, 1:5])
o_central <- original_mw1[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
s_central <- synthetic_mw1[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
o_central$Group <- "Original"
s_central$Group <- "Synthetic"
merged_central <- rbind(o_central, s_central)
merged_central$Region <- "Central"
merged_central$Value <- rowMeans(merged_central[, 1:5])
o_parietal <- original_mw1[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
s_parietal <- synthetic_mw1[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
o_parietal$Group <- "Original"
s_parietal$Group <- "Synthetic"
merged_parietal <- rbind(o_parietal, s_parietal)
merged_parietal$Region <- "Parietal"
merged_parietal$Value <- rowMeans(merged_parietal[, 1:5])
o_occipital <- original_mw1[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
s_occipital <- synthetic_mw1[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
o_occipital$Group <- "Original"
s_occipital$Group <- "Synthetic"
merged_occipital <- rbind(o_occipital, s_occipital)
merged_occipital$Region <- "Occipital"
merged_occipital$Value <- rowMeans(merged_occipital[, 1:5])
o_temporal <- original_mw1[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
s_temporal <- synthetic_mw1[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
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
  labs(x = "Value", y = "Density", title = "Mental Workload Dataset 1", subtitle =  "(Original vs. Synthetic)") +
  theme(
    axis.title = element_text(size = 16, family="Times New Roman", face="bold"),
    axis.text = element_text(size = 11, family = "Times New Roman", face="bold"),
    plot.title = element_text(family = "Times New Roman", face="bold", size=16, hjust = 0.5),
    plot.subtitle = element_text(family = "Times New Roman", size=10, hjust = 0.5),
    legend.text = element_text(family = "Times New Roman", face="bold", size=12),
    legend.title = element_text(family = "Times New Roman", face="bold", size=14, hjust = 0.5),
    strip.text.y = element_text(family = "Times New Roman", size=12)
  )
merged_mw1 <- merged

#######################################################################################################################################
# Mental Workload 2
#######################################################################################################################################
original_mw2 <- read.csv("Original_Dataset_MW2.csv")
synthetic_mw2 <- read.csv("Synthetic_Dataset_MW2.csv")
o_frontal <- original_mw2[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
s_frontal <- synthetic_mw2[,c("Alpha.Frontal", "Beta.Frontal", "Delta.Frontal", "Theta.Frontal", "Gamma.Frontal")]
o_frontal$Group <- "Original"
s_frontal$Group <- "Synthetic"
merged_frontal <- rbind(o_frontal, s_frontal)
merged_frontal$Region <- "Frontal"
merged_frontal$Value <- rowMeans(merged_frontal[, 1:5])
o_central <- original_mw2[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
s_central <- synthetic_mw2[,c("Alpha.Central", "Beta.Central", "Delta.Central", "Theta.Central", "Gamma.Central")]
o_central$Group <- "Original"
s_central$Group <- "Synthetic"
merged_central <- rbind(o_central, s_central)
merged_central$Region <- "Central"
merged_central$Value <- rowMeans(merged_central[, 1:5])
o_parietal <- original_mw2[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
s_parietal <- synthetic_mw2[,c("Alpha.Parietal", "Beta.Parietal", "Delta.Parietal", "Theta.Parietal", "Gamma.Parietal")]
o_parietal$Group <- "Original"
s_parietal$Group <- "Synthetic"
merged_parietal <- rbind(o_parietal, s_parietal)
merged_parietal$Region <- "Parietal"
merged_parietal$Value <- rowMeans(merged_parietal[, 1:5])
o_occipital <- original_mw2[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
s_occipital <- synthetic_mw2[,c("Alpha.Occipital", "Beta.Occipital", "Delta.Occipital", "Theta.Occipital", "Gamma.Occipital")]
o_occipital$Group <- "Original"
s_occipital$Group <- "Synthetic"
merged_occipital <- rbind(o_occipital, s_occipital)
merged_occipital$Region <- "Occipital"
merged_occipital$Value <- rowMeans(merged_occipital[, 1:5])
o_temporal <- original_mw2[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
s_temporal <- synthetic_mw2[,c("Alpha.Temporal", "Beta.Temporal", "Delta.Temporal", "Theta.Temporal", "Gamma.Temporal")]
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
  labs(x = "Value", y = "Density", title = "Mental Workload Dataset 2", subtitle =  "(Original vs. Synthetic)") +
  theme(
    axis.title = element_text(size = 16, family="Times New Roman", face="bold"),
    axis.text = element_text(size = 11, family = "Times New Roman", face="bold"),
    plot.title = element_text(family = "Times New Roman", face="bold", size=16, hjust = 0.5),
    plot.subtitle = element_text(family = "Times New Roman", size=10, hjust = 0.5),
    legend.text = element_text(family = "Times New Roman", face="bold", size=12),
    legend.title = element_text(family = "Times New Roman", face="bold", size=14, hjust = 0.5),
    strip.text.y = element_text(family = "Times New Roman", size=12)
  )
merged_mw2 <- merged
