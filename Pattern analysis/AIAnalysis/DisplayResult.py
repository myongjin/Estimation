import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autoencoder_model import Conv3DVAE

# Ensure parent directory is in sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import PattenAnalysisTool as pat

# Load sequence data
file_name = "AllSequences"
loaded_sequence_data = pat.load_sequences(file_name + '.pkl')
sequences = loaded_sequence_data['sequences']
total_nbSequence = loaded_sequence_data['total_nbSequence']

# Pad sequences to have the same length
max_length = max(len(seq) for seq in sequences)
padded_sequences = np.zeros((total_nbSequence, max_length, 6))
for i, seq in enumerate(sequences):
    padded_sequences[i, :len(seq)] = seq

# Reshape sequences for 3D CNN
sequences_reshaped = np.reshape(padded_sequences, (total_nbSequence, max_length, 2, 3))
sequences_reshaped = np.transpose(sequences_reshaped, (0, 2, 1, 3))
sequences_reshaped = np.expand_dims(sequences_reshaped, axis=-2)
sequences_tensor = torch.tensor(sequences_reshaped, dtype=torch.float32)

# Load the pre-trained model
model_name = 'conv3dvae_model'
model_path = f"{model_name}.pth"
model = Conv3DVAE(input_channels=2, hidden_dims=[32, 64, 128], latent_dim=64, max_length=max_length)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
sequences_tensor = sequences_tensor.to(device)

# Encode sequences to latent vectors
latent_vectors = []
for seq_tensor in sequences_tensor:
    seq_tensor = seq_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        mu, _ = model.encode(seq_tensor)
        latent_vector = mu.cpu().numpy().flatten()
    latent_vectors.append(latent_vector)

# Define the threshold for grouping patterns
threshold = 0.5

# Group sequences based on their latent vectors
groups = {}
for i, latent_vector in enumerate(latent_vectors):
    assigned = False
    for group_id, group_data in groups.items():
        group_vectors = [v for _, v in group_data]
        avg_vector = np.mean(group_vectors, axis=0)
        if np.linalg.norm(latent_vector - avg_vector) < threshold:
            groups[group_id].append((i, latent_vector))
            assigned = True
            break
    if not assigned:
        groups[len(groups)] = [(i, latent_vector)]

# Define colors for each group
colors = plt.cm.get_cmap('tab10', len(groups))

# Plot sequences in each group in separate figures with two subplots
for group_id, group_data in groups.items():
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for seq_idx, _ in group_data:
        seq = sequences[seq_idx]
        ax1.plot(seq[:, 0], seq[:, 1], seq[:, 2], color=colors(group_id))
        ax2.plot(seq[:, 3], seq[:, 4], seq[:, 5], color=colors(group_id))
    ax1.set_title(f"Group {group_id} - Finger 1")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax2.set_title(f"Group {group_id} - Finger 2")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    plt.tight_layout()
    plt.show()
