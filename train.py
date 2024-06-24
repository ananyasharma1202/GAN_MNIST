import os
import pickle
import imageio
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import cv2

from models.generator import Generator
from models.discriminator import Discriminator
from utils.visualization import show_result, show_train_hist
from utils.training import train

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Training parameters
batch_size = config['training']['batch_size']
lr = config['training']['lr']
train_epoch = config['training']['train_epoch']
save_dir = config['training']['save_dir']
log_file = config['training']['log_file']

# Data parameters
dataset = config['data']['dataset']
data_dir = config['data']['data_dir']
image_size = config['data']['image_size']
num_channels = config['data']['num_channels']
normalize_mean = config['data']['normalize_mean']
normalize_std = config['data']['normalize_std']

# Model parameters
generator_input_size = config['model']['generator_input_size']
generator_output_size = config['model']['generator_output_size']
discriminator_input_size = config['model']['discriminator_input_size']
discriminator_output_size = config['model']['discriminator_output_size']

# Data loader
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=num_channels),
    transforms.ToTensor(),
    transforms.Normalize(mean=(normalize_mean,), std=(normalize_std,))
])

if dataset == 'MNIST':
    train_loader = DataLoader(
        datasets.MNIST(data_dir, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Network
G = Generator(input_size=generator_input_size, n_class=generator_output_size).to(device)
D = Discriminator(input_size=discriminator_input_size, n_class=discriminator_output_size).to(device)

# Loss function
BCE_loss = nn.BCELoss()

# Optimizers
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# Results save folder
os.makedirs(f'{save_dir}/Random_results', exist_ok=True)
os.makedirs(f'{save_dir}/Fixed_results', exist_ok=True)

# Fixed noise for visualization
fixed_z_ = torch.randn((5 * 5, generator_input_size)).to(device)

# Open log file
with open(log_file, 'w') as log_f:
    log_f.write("Training Logs\n")

    # Train the model
    generator, train_hist = train(D, G, D_optimizer, G_optimizer, BCE_loss, train_loader, train_epoch, fixed_z_, log_f, config)

    # Save the results
    log_f.write("Training finished! Saving training results...\n")
    torch.save(generator.state_dict(), f"{save_dir}/generator_param.pkl")

    with open(f'{save_dir}/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=f'{save_dir}/MNIST_GAN_train_hist.png')

    # Create a GIF of generated images
    images = []
    for e in range(train_epoch):
        img_name = f'{save_dir}/Fixed_results/MNIST_GAN_{e + 1}.png'
        images.append(cv2.imread(img_name))
    imageio.mimsave(f'{save_dir}/generation_animation.gif', images, fps=5)
