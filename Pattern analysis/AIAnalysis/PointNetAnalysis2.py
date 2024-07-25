import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Custom Max Pooling Layer with Mask
class MaskedMaxPooling2D(nn.Module):
    def forward(self, x, mask):
        x = x.masked_fill(~mask, float('-inf'))
        return torch.max(x, dim=2)[0]

class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (1, 6))  # (batch_size, 1, max_length, 6) -> (batch_size, 32, max_length, 1)
        self.conv2 = nn.Conv2d(32, 64, (1, 1))  # (batch_size, 32, max_length, 1) -> (batch_size, 64, max_length, 1)
        self.conv3 = nn.Conv2d(64, 128, (1, 1))  # (batch_size, 64, max_length, 1) -> (batch_size, 128, max_length, 1)
        self.conv4 = nn.Conv2d(128, 256, (1, 1))  # (batch_size, 128, max_length, 1) -> (batch_size, 256, max_length, 1)
        self.masked_maxpool = MaskedMaxPooling2D()  # (batch_size, 256, max_length, 1) -> (batch_size, 256)
        self.fc1 = nn.Linear(256, 128)  # (batch_size, 256) -> (batch_size, 128)
        self.fc2 = nn.Linear(128, 64)  # (batch_size, 128) -> (batch_size, 64)

    def forward(self, x, mask):
        x = x.unsqueeze(1).to(device)  # (batch_size, max_length, 6) -> (batch_size, 1, max_length, 6)
        mask = mask.to(device)
        x = torch.relu(self.conv1(x))  # (batch_size, 1, max_length, 6) -> (batch_size, 32, max_length, 1)
        x = torch.relu(self.conv2(x))  # (batch_size, 32, max_length, 1) -> (batch_size, 64, max_length, 1)
        x = torch.relu(self.conv3(x))  # (batch_size, 64, max_length, 1) -> (batch_size, 128, max_length, 1)
        x = torch.relu(self.conv4(x))  # (batch_size, 128, max_length, 1) -> (batch_size, 256, max_length, 1)
        x = self.masked_maxpool(x, mask)  # (batch_size, 256, max_length, 1) -> (batch_size, 256)
        x = x.view(x.size(0), -1)  # (batch_size, 256) -> (batch_size, 256)
        x = torch.relu(self.fc1(x))  # (batch_size, 256) -> (batch_size, 128)
        x = self.fc2(x)  # (batch_size, 128) -> (batch_size, 64)
        return x

class Autoencoder(nn.Module):
    def __init__(self, max_length):
        super(Autoencoder, self).__init__()
        self.encoder = PointNetEncoder()  # (batch_size, max_length, 6) -> (batch_size, 64)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),  # (batch_size, 64) -> (batch_size, 128)
            nn.ReLU(True),
            nn.Linear(128, 256),  # (batch_size, 128) -> (batch_size, 256)
            nn.ReLU(True),
            nn.Linear(256, max_length * 6),  # (batch_size, 256) -> (batch_size, max_length * 6)
            nn.Tanh()
        )
        self.max_length = max_length

    def forward(self, x, mask):
        x = self.encoder(x, mask)  # (batch_size, max_length, 6) -> (batch_size, 64)
        x = self.decoder(x)  # (batch_size, 64) -> (batch_size, max_length * 6)
        x = x.view(x.size(0), self.max_length, 6)  # (batch_size, max_length * 6) -> (batch_size, max_length, 6)
        return x

class UNetEncoder(nn.Module):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, padding=1)  # (batch_size, 6, max_length) -> (batch_size, 32, max_length)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # (batch_size, 32, max_length) -> (batch_size, 64, max_length)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # (batch_size, 64, max_length//2) -> (batch_size, 128, max_length//2)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # (batch_size, 128, max_length//4) -> (batch_size, 256, max_length//4)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 다운샘플링 레이어, 길이를 반으로 줄임

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, max_length, 6) -> (batch_size, 6, max_length)
        x1 = torch.relu(self.conv1(x))  # (batch_size, 6, max_length) -> (batch_size, 32, max_length)
        x2 = self.pool(x1)  # (batch_size, 32, max_length) -> (batch_size, 32, max_length//2)
        x2 = torch.relu(self.conv2(x2))  # (batch_size, 32, max_length//2) -> (batch_size, 64, max_length//2)
        x3 = self.pool(x2)  # (batch_size, 64, max_length//2) -> (batch_size, 64, max_length//4)
        x3 = torch.relu(self.conv3(x3))  # (batch_size, 64, max_length//4) -> (batch_size, 128, max_length//4)
        x4 = self.pool(x3)  # (batch_size, 128, max_length//4) -> (batch_size, 128, max_length//8)
        x4 = torch.relu(self.conv4(x4))  # (batch_size, 128, max_length//8) -> (batch_size, 256, max_length//8)
        return x1, x2, x3, x4

class UNetDecoder(nn.Module):
    def __init__(self, max_length):
        super(UNetDecoder, self).__init__()
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)  # (batch_size, 256, max_length//8) -> (batch_size, 128, max_length//4)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)  # (batch_size, 256, max_length//4) -> (batch_size, 128, max_length//4)
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)  # (batch_size, 128, max_length//4) -> (batch_size, 64, max_length//2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)  # (batch_size, 128, max_length//2) -> (batch_size, 64, max_length//2)
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)  # (batch_size, 64, max_length//2) -> (batch_size, 32, max_length)
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)  # (batch_size, 64, max_length) -> (batch_size, 32, max_length)
        self.final_conv = nn.Conv1d(32, 6, kernel_size=1)  # (batch_size, 32, max_length) -> (batch_size, 6, max_length)
        self.max_length = max_length

    def forward(self, x1, x2, x3, x4):
        x = self.upconv3(x4)  # (batch_size, 256, max_length//8) -> (batch_size, 128, max_length//4)
        x = torch.cat((x, x3[:, :, :x.size(2)]), dim=1)  # 패딩에 따른 크기 불일치를 맞추기 위해 슬라이싱
        x = torch.relu(self.conv3(x))  # (batch_size, 256, max_length//4) -> (batch_size, 128, max_length//4)
        x = self.upconv2(x)  # (batch_size, 128, max_length//4) -> (batch_size, 64, max_length//2)
        x = torch.cat((x, x2[:, :, :x.size(2)]), dim=1)  # 패딩에 따른 크기 불일치를 맞추기 위해 슬라이싱
        x = torch.relu(self.conv2(x))  # (batch_size, 128, max_length//2) -> (batch_size, 64, max_length//2)
        x = self.upconv1(x)  # (batch_size, 64, max_length//2) -> (batch_size, 32, max_length)
        x = torch.cat((x, x1[:, :, :x.size(2)]), dim=1)  # 패딩에 따른 크기 불일치를 맞추기 위해 슬라이싱
        x = torch.relu(self.conv1(x))  # (batch_size, 64, max_length) -> (batch_size, 32, max_length)
        x = self.final_conv(x)  # (batch_size, 32, max_length) -> (batch_size, 6, max_length)
        return x

class UNetAutoencoder(nn.Module):
    def __init__(self, max_length):
        super(UNetAutoencoder, self).__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder(max_length)

    def forward(self, x, mask):
        x1, x2, x3, x4 = self.encoder(x.transpose(1, 2))  # (batch_size, max_length, 6) -> (batch_size, 6, max_length)
        x = self.decoder(x1, x2, x3, x4)
        return x.transpose(1, 2)  # (batch_size, 6, max_length) -> (batch_size, max_length, 6)


# 모델 로드 여부를 결정하는 플래그
load_model = False
model_path = 'UnetAutoEncoder_Model_Conv2D_twoPos.pth'
start_epoch = 0
epochs = 100
save_interval = 10

# Load data
with open('ID002_Sequence.pkl', 'rb') as f:
    data = pickle.load(f)

sequences = data['sequences']

# Find the maximum length
max_length = max([seq.shape[0] for seq in sequences])
print("Maximum length:", max_length)

# Pad sequences to the same length and create initial mask
padded_sequences = []
masks = []
for seq in sequences:
    if seq.shape[0] < max_length:
        pad_width = max_length - seq.shape[0]
        padded_seq = np.pad(seq, ((0, pad_width), (0, 0)), 'constant')
        mask = np.concatenate([np.ones(seq.shape[0]), np.zeros(pad_width)])
    else:
        padded_seq = seq
        mask = np.ones(seq.shape[0])
    padded_sequences.append(padded_seq)
    masks.append(mask)

padded_sequences = np.array(padded_sequences)
print("Padded sequences shape:", padded_sequences.shape)

# Normalize data
padded_sequences_min = padded_sequences.min(axis=(0, 1), keepdims=True)
padded_sequences_max = padded_sequences.max(axis=(0, 1), keepdims=True)
padded_sequences = (padded_sequences - padded_sequences_min) / (padded_sequences_max - padded_sequences_min + 1e-8)
print("Normalized padded sequences shape:", padded_sequences.shape)

masks = np.array(masks)
print("Masks shape (before expanding):", masks.shape)

# Convert to PyTorch tensor
padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32).to(device)
masks = torch.tensor(masks, dtype=torch.bool).to(device)

print("Padded sequences tensor shape:", padded_sequences.shape)
print("Masks tensor shape:", masks.shape)

# Expand mask dimensions to match Conv4 output
masks = masks.unsqueeze(1).unsqueeze(3).expand(-1, 256, -1, 1)
print("Masks shape (after expanding):", masks.shape)

#Autoencoder
#model = Autoencoder(max_length).to(device)
#UnetAutoencoder
model = UNetAutoencoder(max_length).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

if load_model and os.path.isfile(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"모델을 불러왔습니다. 시작 에폭: {start_epoch}")
else:
    print("새 모델을 만듭니다.")

# Training loop
losses = []

if start_epoch < epochs:
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for seq, mask in zip(padded_sequences, masks):
            seq = seq.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = model(seq, mask)

            loss = criterion(output, seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(padded_sequences)
        losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, model_path)
            print(f"모델이 {epoch + 1} 에폭에서 저장되었습니다.")

    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, model_path)
    print("최종 모델이 저장되었습니다.")
else:
    print("모델이 이미 모든 에폭에 대해 학습되었습니다.")

# 손실 변화를 시각화하는 함수
def plot_loss_curve(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

plot_loss_curve(losses)

# Extract latent vectors
latent_vectors = []
with torch.no_grad():
    for seq, mask in zip(padded_sequences, masks):
        seq = seq.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        latent_vector = model.encoder(seq, mask)
        latent_vectors.append(latent_vector.squeeze().cpu().numpy())

latent_vectors = np.array(latent_vectors)

# K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(latent_vectors)

plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    cluster = latent_vectors[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i}')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Clustering of Latent Vectors')
plt.legend()
plt.show()

# 각 시퀀스의 클러스터 레이블을 출력
for i, label in enumerate(labels):
    print(f'Sequence {i} is in Cluster {label}')

# 시퀀스 그리기 함수
def plot_cluster_sequences(sequences, labels, cluster_num):
    fig = plt.figure(figsize=(12, 8))

    # 첫 번째 3자유도 데이터 플롯
    ax1 = fig.add_subplot(121, projection='3d')
    for i, seq in enumerate(sequences):
        if labels[i] == cluster_num:
            seq1 = seq[:3, :].T  # 첫 번째 3자유도 데이터
            ax1.plot(seq1[:, 0], seq1[:, 1], seq1[:, 2], label=f'Sequence {i} - Part 1', alpha=0.6)
    ax1.set_title(f'Cluster {cluster_num} Sequences - Part 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 두 번째 3자유도 데이터 플롯
    ax2 = fig.add_subplot(122, projection='3d')
    for i, seq in enumerate(sequences):
        if labels[i] == cluster_num:
            seq2 = seq[3:, :].T  # 두 번째 3자유도 데이터
            ax2.plot(seq2[:, 0], seq2[:, 1], seq2[:, 2], label=f'Sequence {i} - Part 2', alpha=0.6)
    ax2.set_title(f'Cluster {cluster_num} Sequences - Part 2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.legend()
    plt.show()

# 특정 클러스터의 시퀀스 그리기
cluster_to_plot = 2
plot_cluster_sequences(padded_sequences, labels, cluster_to_plot)

def plot_original_vs_reconstructed(original_sequences, model, masks, num_samples=5):
    model.eval()
    indices = random.sample(range(len(original_sequences)), num_samples)

    fig = plt.figure(figsize=(15, num_samples * 5))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            original = original_sequences[idx]
            mask = masks[idx].unsqueeze(0).to(device)
            original_tensor = original.unsqueeze(0).to(device)
            reconstructed_tensor = model(original_tensor, mask).squeeze(0).cpu().numpy()

            # 원본 시퀀스 Part 1
            ax1 = fig.add_subplot(num_samples, 2, 2*i+1, projection='3d')
            ax1.plot(original[:, 0], original[:, 1], original[:, 2], label='Original Part 1', alpha=0.6)
            ax1.plot(reconstructed_tensor[:, 0], reconstructed_tensor[:, 1], reconstructed_tensor[:, 2], label='Reconstructed Part 1', alpha=0.6)
            ax1.set_title(f'Original vs Reconstructed Sequence {idx} - Part 1')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()

            # 원본 시퀀스 Part 2
            ax2 = fig.add_subplot(num_samples, 2, 2*i+2, projection='3d')
            ax2.plot(original[:, 3], original[:, 4], original[:, 5], label='Original Part 2', alpha=0.6)
            ax2.plot(reconstructed_tensor[:, 3], reconstructed_tensor[:, 4], reconstructed_tensor[:, 5], label='Reconstructed Part 2', alpha=0.6)
            ax2.set_title(f'Original vs Reconstructed Sequence {idx} - Part 2')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()

    plt.tight_layout()
    plt.show()

# 원본 시퀀스와 재구성된 시퀀스를 비교하여 플롯
plot_original_vs_reconstructed(padded_sequences, model, masks)

