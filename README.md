# MNIST Autoencoder, Classifier & Task-Specific Encoding

A PyTorch project exploring how the **training objective shapes a learned representation**. Using MNIST, I train convolutional autoencoders, a classifier built on the same encoder architecture, and a decoder trained on top of a *frozen classifier encoder* to see what information a task-specific latent space keeps and what it throws away.

## Highlights

- Convolutional **autoencoder** (encoder + decoder) with configurable latent size and channel width
- **Classifier** that reuses the same encoder, trained on the full dataset vs. only **100 labeled samples**
- **Task-specific encoding**: freeze a classifier-trained encoder and train only a decoder to reconstruct images from it
- Per-digit visualizations to study **in-class reconstruction variability**

## Project Structure

| File | Description |
|------|-------------|
| `Autoencoder.py` | Data loader, `Encoder`, `Decoder`, `CNN_AE`, training/evaluation loops, reconstruction and loss visualizations. Runs a grid over latent dimension (4, 16) and base channels (4, 16). |
| `Classifier.py` | `Classifier` (encoder + linear head). Compares training on the full MNIST train set vs. a 100-sample subset. |
| `Task_specific_encoding.py` | `DecoderOnlyAE`: trains a classifier, freezes its encoder, then trains a decoder to reconstruct inputs from the classifier's latent code. |

## Method

**Architecture**
- Encoder: 3 strided `Conv2d` layers (28x28 padded to 32x32, then 16 -> 8 -> 4), ReLU, followed by a linear layer to the latent vector
- Decoder: linear layer + 3 `ConvTranspose2d` layers with a Sigmoid output, cropped back to 28x28
- Classifier: encoder + linear layer to 10 classes (CrossEntropy loss)

**Experiments**
1. **Autoencoder capacity study**: 4 settings of `latent_dim` x `base_channels`, L1 reconstruction loss, AdamW, 10 epochs
2. **Data efficiency**: classifier trained on 60k vs. 100 samples, 30 epochs, tracking loss and accuracy on the test set; evaluated both with an encoder pretrained as an autoencoder (frozen) and trained from scratch
3. **Task-specific representation**: classifier encoder frozen, decoder trained with L1 loss, then reconstructions inspected per digit class

## Getting Started

```bash
git clone <repo-url>
cd <repo-name>
pip install torch torchvision matplotlib numpy
```

MNIST is downloaded automatically into `./data` on first run.

```bash
python Autoencoder.py            # autoencoder grid + reconstructions
python Classifier.py             # full vs. small-data classifier
python Task_specific_encoding.py # frozen classifier encoder + trained decoder
```

A GPU is used automatically if available.

## Results

**Setup:** an autoencoder (`latent_dim=16`, `base_channels=16`) is trained for 20 epochs with L1 loss (final train loss **0.0278**). Its encoder is then **frozen** and a single linear layer is trained on the 16-dimensional latent code for 30 epochs, using either the full training set or only 100 labeled samples. All numbers are on the 10,000-image MNIST test set.

| Labeled training data | Train acc. | Test acc. | Test loss |
|-----------------------|-----------:|----------:|----------:|
| Full (60,000)         | 92.01%     | **92.64%**| 0.253     |
| 100 samples           | 100.00%    | **76.47%**| 0.859     |

![Frozen-encoder classifier: full dataset vs. 100 samples](images/frozen_encoder_classifier.png)

## Key Takeaways

- An encoder trained **only to reconstruct** already produces a latent space where a linear layer reaches **~92.6%** test accuracy, with no labels used for the encoder.
- With just **100 labeled examples** the same linear head reaches **~76.5%**, compared to ~8% after the first epoch, so the unsupervised representation transfers well to a low-label regime.
- The small-data model fits its training set perfectly (100%) while test accuracy plateaus at ~76%: a clear overfitting gap, as expected with 100 samples.
- On the full set, test accuracy slightly exceeds train accuracy; the linear head is limited by the quality of the reconstruction-oriented features rather than by overfitting. This motivates the task-specific encoder experiment.

## Tech Stack

Python, PyTorch, torchvision, NumPy, Matplotlib

## Author

**Ori** - Computer Science graduate, The Hebrew University of Jerusalem
[LinkedIn](#) | [Email](#)
