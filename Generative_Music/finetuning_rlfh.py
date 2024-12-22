#!/usr/bin/env python3
"""
Combine everything into an end-to-end pipeline
and apply a simplistic RLHF approach.

- We load the trained VAE, Diffusion model, and AccelerometerToLatent model.
- We synthesize short musical segments from random or real accelerometer data.
- We collect 'feedback' (in a placeholder way), train a reward model,
  and perform policy-gradient updates on the generation pipeline.

Note: This script is a conceptual template. Real RLHF would involve
actual user studies, a proper reward model, etc.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vae_model import VAE
from diffusion_decoder import SimpleDiffusionModel
from accelerometer_model import AccelerometerToLatent

class RewardModel(nn.Module):
    """
    Predicts a 'focus-maintenance score' given a generated audio representation.
    In practice, you might input the final audio embedding from a pretrained audio model
    or from the VAE/diffusion's output. Here we do something trivial.
    """
    def __init__(self, input_dim=64):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class RLHFTrainer:
    """
    A very simplified RLHF-like trainer:
    - We treat the combination of (Accelerometer -> Latent -> Diffusion -> 'Audio Embedding')
      as a policy π_\theta.
    - Sample from this policy, get reward from the RewardModel, then do gradient ascent.
    - This is conceptual; real RLHF would likely require advanced technique like PPO.
    """
    def __init__(self, accel_model, diffusion_model, reward_model, device):
        self.accel_model = accel_model
        self.diffusion_model = diffusion_model
        self.reward_model = reward_model
        self.device = device

        # Combine parameters for naive "end-to-end" training
        self.optimizer = optim.Adam(
            list(self.accel_model.parameters()) + list(self.diffusion_model.parameters()),
            lr=1e-4
        )

    def sample_policy(self, acc_features):
        """
        1. accel -> latent
        2. diffusion -> spectrogram (flattened)
        3. Convert spectrogram to some embedding (here, just average pooling as placeholder)
        """
        # Step 1
        latent = self.accel_model(acc_features)
        # Step 2
        spec_flat = self.diffusion_model(latent)
        # Step 3
        # Convert spec_flat (batch_size, 128*128) -> (batch_size, 1) by mean
        audio_embedding = spec_flat.mean(dim=1, keepdim=True)
        return audio_embedding

    def compute_loss(self, audio_embedding):
        # Reward is predicted by the reward model
        reward = self.reward_model(audio_embedding)
        # We want to maximize reward -> minimize negative reward
        loss = -reward.mean()
        return loss

    def train_step(self, acc_features):
        """
        Single batch RL update.
        """
        self.optimizer.zero_grad()
        audio_embedding = self.sample_policy(acc_features)
        loss = self.compute_loss(audio_embedding)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load pre-trained models
    # VAE (if needed for advanced usage). We'll load but not use in this simple pipeline.
    vae = VAE()
    vae.load_state_dict(torch.load("models/vae_music.pt", map_location=device))
    vae.to(device)
    vae.eval()

    # Accelerometer -> Latent
    accel_model = AccelerometerToLatent(input_dim=3, latent_dim=64)
    accel_model.load_state_dict(torch.load("models/acc_to_latent.pt", map_location=device))
    accel_model.to(device)

    # Diffusion decoder
    diffusion_model = SimpleDiffusionModel(latent_dim=64, spec_shape=(128, 128))
    diffusion_model.load_state_dict(torch.load("models/diffusion_decoder.pt", map_location=device))
    diffusion_model.to(device)

    # 2. Reward model initialization
    reward_model = RewardModel(input_dim=1).to(device)  # Because we do a single scalar embedding
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)

    # Dummy "training data" for reward model:
    # We pretend we have some pairs (audio_embedding, focus_score)
    # In practice, gather real human feedback via a user interface.
    num_dummy_samples = 100
    dummy_audio_embeddings = torch.rand(num_dummy_samples, 1).to(device)
    dummy_focus_scores = torch.rand(num_dummy_samples, 1).to(device)

    # 2a. Train reward model (supervised) from dummy data
    for epoch in range(3):
        idxs = torch.randperm(num_dummy_samples)
        for i in range(0, num_dummy_samples, 8):
            batch_idx = idxs[i : i+8]
            x_batch = dummy_audio_embeddings[batch_idx]
            y_batch = dummy_focus_scores[batch_idx]

            reward_optimizer.zero_grad()
            preds = reward_model(x_batch)
            loss = nn.functional.mse_loss(preds, y_batch)
            loss.backward()
            reward_optimizer.step()

    print("Reward model dummy training complete.")

    # 3. RLHF-like Fine-Tuning
    rlhf_trainer = RLHFTrainer(accel_model, diffusion_model, reward_model, device)

    # Create dummy accelerometer inputs
    # In real usage, load from data or stream in real-time
    dummy_acc_data = torch.rand(64, 3).to(device)  # batch of 64

    for step in range(10):
        loss = rlhf_trainer.train_step(dummy_acc_data)
        print(f"RL step {step}, loss (negative reward): {loss:.4f}")

    # 4. Save updated models
    os.makedirs("models", exist_ok=True)
    torch.save(accel_model.state_dict(), "models/acc_to_latent_rlhf.pt")
    torch.save(diffusion_model.state_dict(), "models/diffusion_decoder_rlhf.pt")
    print("RLHF fine-tuning completed and models updated.")

if __name__ == "__main__":
    main()
