import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from Autoencoder import MNISTDataLoader, Encoder, Decoder, visualize_reconstruction
from Classifier import train_classifier, evaluate_classifier, Classifier

torch.manual_seed(0)

# ==============================================================================
# Model Definition
# ==============================================================================
class DecoderOnlyAE(nn.Module):
    """
    Autoencoder variant that uses a pretrained, frozen encoder.
    Only the decoder is trained to reconstruct input images.
    """
    def __init__(self, pretrained_encoder, latent_dim=16, base_channels=4):
        super().__init__()
        self.encoder = pretrained_encoder
        self.decoder = Decoder(latent_dim, base_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ==============================================================================
# Training and Evaluation
# ==============================================================================
def freeze_encoder(model):
    """Freeze encoder parameters (no gradient updates)."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    return model


def train(model, dataloader, optimizer, criterion, device):
    """Train decoder-only autoencoder for one epoch."""
    model.train()
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate reconstruction loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            total_loss += criterion(model(images), images).item()
    return total_loss / len(dataloader)


def plot_loss(train_losses, test_losses):
    """Plot training and test loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='x')
    plt.title("Decoder-only Autoencoder Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Reconstruction Analysis
# ==============================================================================
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

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs_cls = 2
    epochs_dec = 2
    latent_dim = 16
    base_channels = 16

    # Load Data
    mnist_loader = MNISTDataLoader(batch_size=batch_size)
    train_loader, test_loader = mnist_loader.train_loader, mnist_loader.test_loader

    # Train Classifier Encoder
    encoder = Encoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
    classifier = Classifier(encoder, latent_dim=latent_dim, num_classes=10).to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training Classifier Encoder:")
    for epoch in range(epochs_cls):
        tr_loss, tr_acc = train_classifier(classifier, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate_classifier(classifier, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs_cls} | Train Acc: {tr_acc:.2f}% | Test Acc: {te_acc:.2f}%")

    pretrained_encoder = classifier.get_encoder()

    # Train Decoder for Reconstruction
    model = DecoderOnlyAE(pretrained_encoder, latent_dim, base_channels).to(device)
    freeze_encoder(model)
    optimizer = optim.AdamW(model.decoder.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    print("\nTraining Decoder for Reconstruction:")
    train_losses, test_losses = [], []
    for epoch in range(epochs_dec):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{epochs_dec} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    visualize_reconstruction(model, test_loader, device)
    plot_loss(train_losses, test_losses)

    # Analyze In-Class Reconstruction Variability
    print("\nAnalyzing in-class reconstruction variability...")
    analyze_in_class_variability(model, test_loader, device)
