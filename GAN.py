import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Original_Dataset_Stress.csv")
data.drop(columns=["Subject"], inplace=True)
features = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]
target = "Condition"

data_tensor = torch.tensor(data[features].values, dtype=torch.float32)
labels_tensor = torch.tensor(data[target].values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(data_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define GAN model
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


input_dim = len(features)
latent_dim = 16
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)
optimizer_G = optim.Adam(generator.parameters(), lr=1e-3)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-3)
criterion = nn.BCELoss()
epochs = 50
generator.train()
discriminator.train()


for epoch in range(epochs):
    for real_data, _ in dataloader:
        batch_size = real_data.size(0)
        
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        real_loss = criterion(discriminator(real_data), real_labels)
        
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_data), real_labels)
        g_loss.backward()
        optimizer_G.step()
    
    print(f"Epoch {epoch+1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")


generator.eval()
with torch.no_grad():
    z = torch.randn((len(data), latent_dim))
    synthetic_data = generator(z).numpy()

synthetic_df = pd.DataFrame(synthetic_data, columns=features)
synthetic_df[target] = 1
real_df = data.copy()
real_df[target] = 0
combined_df = pd.concat([real_df, synthetic_df])
labels = np.array(combined_df[target])
combined_df.drop(columns=[target], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(combined_df, labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
for i, column in enumerate(features):
    sns.kdeplot(real_df[column], label="Original", ax=axes[i])
    sns.kdeplot(synthetic_df[column], label="Synthetic", ax=axes[i])
    axes[i].set_title(column)
    axes[i].legend()
plt.tight_layout()
plt.show()

synthetic_df.to_csv("synthetic_EEG_data_GAN.csv", index=False)
# Random Forest Accuracy: 0.9883