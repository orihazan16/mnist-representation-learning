import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from Autoencoder import MNISTDataLoader, Encoder

torch.manual_seed(0)

# ==============================
# Classifier definition
# ==============================
class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim=16, num_classes=10):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(latent_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_probs=False):
        z = self.encoder(x)
        logits = self.fc(z)
        if return_probs:
            return self.softmax(logits)
        return logits

    def get_encoder(self):
        return self.encoder

# ==============================
# Training & Evaluation
# ==============================
def train_classifier(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100. * correct / total

def evaluate_classifier(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), 100. * correct / total

# ==============================
# Experiment setup
# ==============================
def run_training(full_train_loader, small_train_loader, test_loader, device, epochs=30, latent_dim=16, base_channels=16):
    encoder_full = Encoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
    encoder_small = Encoder(latent_dim=latent_dim, base_channels=base_channels).to(device)

    classifier_full = Classifier(encoder_full, latent_dim).to(device)
    classifier_small = Classifier(encoder_small, latent_dim).to(device)

    optimizer_full = optim.AdamW(classifier_full.parameters(), lr=1e-4)
    optimizer_small = optim.AdamW(classifier_small.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses_full, test_losses_full, train_accs_full, test_accs_full = [], [], [], []
    train_losses_small, test_losses_small, train_accs_small, test_accs_small = [], [], [], []

    for epoch in range(epochs):
        # Full dataset
        tr_loss, tr_acc = train_classifier(classifier_full, full_train_loader, optimizer_full, criterion, device)
        te_loss, te_acc = evaluate_classifier(classifier_full, test_loader, criterion, device)
        train_losses_full.append(tr_loss); test_losses_full.append(te_loss)
        train_accs_full.append(tr_acc); test_accs_full.append(te_acc)

        # Small dataset
        tr_loss_s, tr_acc_s = train_classifier(classifier_small, small_train_loader, optimizer_small, criterion, device)
        te_loss_s, te_acc_s = evaluate_classifier(classifier_small, test_loader, criterion, device)
        train_losses_small.append(tr_loss_s); test_losses_small.append(te_loss_s)
        train_accs_small.append(tr_acc_s); test_accs_small.append(te_acc_s)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Full -> Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}%, Test Loss: {te_loss:.4f}, Acc: {te_acc:.2f}%")
        print(f"Small -> Train Loss: {tr_loss_s:.4f}, Acc: {tr_acc_s:.2f}%, Test Loss: {te_loss_s:.4f}, Acc: {te_acc_s:.2f}%\n")

    return (train_losses_full, train_accs_full, test_losses_full, test_accs_full,
            train_losses_small, train_accs_small, test_losses_small, test_accs_small)

# ==============================
# Plot results
# ==============================
def plot_results(train_losses_full, test_losses_full, train_accs_full, test_accs_full,
                 train_losses_small, test_losses_small, train_accs_small, test_accs_small):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(train_losses_full, label='Train (Full)')
    axes[0, 0].plot(test_losses_full, label='Test (Full)')
    axes[0, 0].set_title("Loss - Full Dataset")
    axes[0, 0].legend(); axes[0, 0].grid()

    axes[0, 1].plot(train_accs_full, label='Train (Full)')
    axes[0, 1].plot(test_accs_full, label='Test (Full)')
    axes[0, 1].set_title("Accuracy - Full Dataset")
    axes[0, 1].legend(); axes[0, 1].grid()

    axes[1, 0].plot(train_losses_small, label='Train (100 samples)')
    axes[1, 0].plot(test_losses_small, label='Test (100 samples)')
    axes[1, 0].set_title("Loss - Small Dataset")
    axes[1, 0].legend(); axes[1, 0].grid()

    axes[1, 1].plot(train_accs_small, label='Train (100 samples)')
    axes[1, 1].plot(test_accs_small, label='Test (100 samples)')
    axes[1, 1].set_title("Accuracy - Small Dataset")
    axes[1, 1].legend(); axes[1, 1].grid()

    plt.tight_layout()
    plt.show()

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 30
    latent_dim = 16
    base_channels = 16

    # Load data
    mnist_loader = MNISTDataLoader(batch_size=batch_size)
    full_train_loader = mnist_loader.train_loader
    test_loader = mnist_loader.test_loader

    # Small 100-sample subset
    dataset_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    subset = data_utils.Subset(dataset_full, torch.arange(100))
    small_train_loader = data_utils.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Run training
    results = run_training(full_train_loader, small_train_loader, test_loader, device,
                           epochs=epochs, latent_dim=latent_dim, base_channels=base_channels)

    plot_results(*results)