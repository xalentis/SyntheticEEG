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

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


input_dim = len(features)
latent_dim = 16
vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = criterion(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

epochs = 50
vae.train()

for epoch in range(epochs):
    for real_data, _ in dataloader:
        optimizer.zero_grad()
        recon_data, mu, logvar = vae(real_data)
        loss = vae_loss(recon_data, real_data, mu, logvar)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Generate synthetic EEG data
vae.eval()
with torch.no_grad():
    z = torch.randn((len(data), latent_dim))
    synthetic_data = vae.decoder(z).numpy()

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

synthetic_df.to_csv("synthetic_EEG_data_VAE.csv", index=False)
# Random Forest Accuracy: 0.9708