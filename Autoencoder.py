import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

# ==============================
# Data Loader
# ==============================
class MNISTDataLoader:
    def __init__(self, batch_size=32):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_loader = self._create_loader(train=True, batch_size=batch_size)
        self.test_loader = self._create_loader(train=False, batch_size=batch_size)

    def _create_loader(self, train, batch_size):
        dataset = datasets.MNIST(
            root="data", train=train, download=True, transform=self.transform
        )
        return data_utils.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

# ==============================
# Encoder
# ==============================
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.ConstantPad2d(2, 0),  # 28x28 -> 32x32
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1),      # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1), # 8 -> 4
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4*4*base_channels*4, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

# ==============================
# Decoder
# ==============================
class Decoder(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super(Decoder, self).__init__()
        self.base_channels = base_channels
        self.fc = nn.Linear(latent_dim, 4*4*base_channels*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels*4, 4, 4)
        x = self.deconv(x)
        return x[:, :, 2:-2, 2:-2]  # crop back to 28x28

# ==============================
# Autoencoder
# ==============================
class CNN_AE(nn.Module):
    def __init__(self, latent_dim=16, base_channels=4):
        super(CNN_AE, self).__init__()
        self.encoder = Encoder(latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, base_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ==============================
# Training and Evaluation
# ==============================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, _ in dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ==============================
# Visualization
# ==============================
def visualize_reconstruction(model, dataloader, device, n_images=5):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)
    outputs = model(images)
    images, outputs = images.cpu().detach(), outputs.cpu().detach()

    fig, axes = plt.subplots(2, n_images, figsize=(12, 4))
    for i in range(n_images):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstruction')
    plt.show()

def plot_losses(loss_histories, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (setting, history) in enumerate(loss_histories.items()):
        ax = axes[idx]
        train_losses, val_losses = history
        ax.plot(range(1, epochs+1), train_losses, label='Train Loss')
        ax.plot(range(1, epochs+1), val_losses, label='Test Loss')
        ax.set_title(f"Latent {setting[0]}, BaseChannels {setting[1]}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def collect_digit_examples(model, dataloader, device, num_per_digit=10):
    """
    Collect original and reconstructed samples per digit class.
    Used for visual inspection and in-class variability analysis.
    """
    model.eval()
    examples = {i: {'orig': [], 'recon': []} for i in range(10)}
    counts = {i: 0 for i in range(10)}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            recon = model(images)
            for i in range(images.size(0)):
                d = labels[i].item()
                if counts[d] < num_per_digit:
                    examples[d]['orig'].append(images[i].cpu().numpy())
                    examples[d]['recon'].append(recon[i].cpu().numpy())
                    counts[d] += 1
            if all(c >= num_per_digit for c in counts.values()):
                break

    for d in examples:
        examples[d]['orig'] = np.array(examples[d]['orig'])
        examples[d]['recon'] = np.array(examples[d]['recon'])
    return examples


def visualize_digit_reconstructions(examples, num_per_digit=10):
    """Display per-digit rows of original and reconstructed samples."""
    plt.figure(figsize=(2 * num_per_digit, 20))
    for d in range(10):
        for i in range(num_per_digit):
            plt.subplot(20, num_per_digit, 2 * d * num_per_digit + i + 1)
            plt.imshow(examples[d]['orig'][i].reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.subplot(20, num_per_digit, 2 * d * num_per_digit + num_per_digit + i + 1)
            plt.imshow(examples[d]['recon'][i].reshape(28, 28), cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def analyze_in_class_variability(model, loader, device, num_per_digit=10):
    """Visualize multiple examples per digit class to observe reconstruction variety."""
    examples = collect_digit_examples(model, loader, device, num_per_digit)
    visualize_digit_reconstructions(examples)


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    mnist_loader = MNISTDataLoader(batch_size=batch_size)

    settings = [
        {'latent_dim': 4, 'base_channels': 4},
        {'latent_dim': 16, 'base_channels': 4},
        {'latent_dim': 4, 'base_channels': 16},
        {'latent_dim': 16, 'base_channels': 16},
    ]

    epochs = 10
    loss_histories = {}

    for setting in settings:
        print(f"\nTraining AE with latent_dim={setting['latent_dim']} and base_channels={setting['base_channels']}")
        model = CNN_AE(latent_dim=setting['latent_dim'], base_channels=setting['base_channels']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss()

        train_losses, val_losses = [], []
        for epoch in range(1, epochs + 1):
            train_loss = train(model, mnist_loader.train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, mnist_loader.test_loader, criterion, device)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {val_loss:.4f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        loss_histories[(setting['latent_dim'], setting['base_channels'])] = (train_losses, val_losses)
        visualize_reconstruction(model, mnist_loader.test_loader, device)

    plot_losses(loss_histories, epochs)
