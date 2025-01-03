import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer


# Define a basic GAN generator
class Generator(nn.Module):
    def __init__(self, latent_dim, text_dim, image_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size * image_size * 3),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        x = torch.cat([noise, text_embedding], dim=1)
        x = self.fc(x)
        return x.view(x.size(0), 3, image_size, image_size)




# unction to query the model

# Define a basic GAN discriminator
class Discriminator(nn.Module):
    def __init__(self, text_dim, image_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_size * image_size * 3, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding):
        image_flat = image.view(image.size(0), -1)
        x = torch.cat([image_flat, text_embedding], dim=1)
        return self.fc(x)


# Training loop
def train(generator, discriminator, dataloader, text_model, optimizer_G, optimizer_D, criterion, latent_dim, device):
    for epoch in range(num_epochs):
        for i, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)

            # Generate noise and text embeddings
            noise = torch.randn(images.size(0), latent_dim).to(device)
            text_embeddings = text_model(captions).last_hidden_state.mean(dim=1)

            # Generate fake images
            fake_images = generator(noise, text_embeddings)

            # Train discriminator
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(images, text_embeddings), real_labels)
            fake_loss = criterion(discriminator(fake_images.detach(), text_embeddings), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_images, text_embeddings), real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


# Parameters
latent_dim = 100
text_dim = 512  # CLIP text embedding dimension
image_size = 64
batch_size = 64
num_epochs = 100
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
generator = Generator(latent_dim, text_dim, image_size).to(device)
discriminator = Discriminator(text_dim, image_size).to(device)

# Optimizers and Loss
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Text Model (CLIP tokenizer and encoder)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Data Loader (dummy dataset for simplicity)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.FakeData(transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train
train(generator, discriminator, dataloader, text_model, optimizer_G, optimizer_D, criterion, latent_dim, device)

