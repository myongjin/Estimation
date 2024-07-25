import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.cluster import KMeans
import os


def square_distance(src, dst):
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def masked_max_pooling(x, mask):
    # x: (batch_size, channels, num_points)
    # mask: (batch_size, 1, num_points)
    x_masked = x * mask + (1 - mask) * torch.finfo(x.dtype).min
    pooled, _ = torch.max(x_masked, dim=2)
    return pooled


def sample_and_group(npoint, radius, nsample, xyz, points):
    B, C, N = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz.transpose(2, 1).contiguous(), npoint)  # [B, npoint]
    new_xyz = index_points(xyz.transpose(2, 1).contiguous(), fps_idx)
    new_xyz = new_xyz.transpose(2, 1).contiguous()

    idx = query_ball_point(radius, nsample, xyz.transpose(2, 1).contiguous(), new_xyz.transpose(2, 1).contiguous())
    grouped_xyz = index_points(xyz.transpose(2, 1).contiguous(), idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.transpose(2, 1).unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points.transpose(2, 1).contiguous(), idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points.permute(0, 3, 2, 1)


def sample_and_group_all(xyz, points):
    B, N, C = xyz.shape
    new_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, mask):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = masked_max_pooling(new_points, mask)
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2Encoder(nn.Module):
    def __init__(self, feature_dim=128, input_channels=6):
        super(PointNet2Encoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=input_channels, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, xyz, mask):
        print(f"Encoder input shape: {xyz.shape}")
        print(f"Encoder mask shape: {mask.shape}")
        B, C, N = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz[:, :3, :], xyz, mask)  # Only use first 3 channels for xyz
        print(f"SA1 output shape: xyz {l1_xyz.shape}, points {l1_points.shape}")
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, mask)
        print(f"SA2 output shape: xyz {l2_xyz.shape}, points {l2_points.shape}")
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, mask)
        print(f"SA3 output shape: xyz {l3_xyz.shape}, points {l3_points.shape}")
        x = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        print(f"Encoder final output shape: {x.shape}")
        return x

class PointNet2Decoder(nn.Module):
    def __init__(self, feature_dim=128, num_points=1024):
        super(PointNet2Decoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_points * 3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, self.num_points)
        return x


class PointNet2AutoEncoder(nn.Module):
    def __init__(self, feature_dim=128, num_points=1024, input_channels=6):
        super(PointNet2AutoEncoder, self).__init__()
        self.encoder = PointNet2Encoder(feature_dim, input_channels)
        self.decoder = PointNet2Decoder(feature_dim, num_points)

    def forward(self, x, mask):
        print(f"AutoEncoder input shape: {x.shape}")
        print(f"AutoEncoder mask shape: {mask.shape}")
        x = x.transpose(1, 2)  # [B, C, N]으로 변경
        mask = mask.unsqueeze(1)  # [B, 1, N]으로 변경
        print(f"AutoEncoder transposed input shape: {x.shape}")
        print(f"AutoEncoder unsqueezed mask shape: {mask.shape}")
        features = self.encoder(x, mask)
        print(f"Encoder output shape: {features.shape}")
        reconstructed = self.decoder(features)
        print(f"Decoder output shape: {reconstructed.shape}")
        return reconstructed, features


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.max_length = max(len(seq) for seq in sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        padded_sequence = np.zeros((self.max_length, sequence.shape[1]), dtype=np.float32)
        padded_sequence[:len(sequence)] = sequence
        mask = np.zeros(self.max_length, dtype=np.float32)
        mask[:len(sequence)] = 1
        return torch.FloatTensor(padded_sequence), torch.FloatTensor(mask)

def custom_collate(batch):
    sequences, masks = zip(*batch)
    return torch.stack(sequences), torch.stack(masks)


def load_sequences(file_name):
    with open(file_name, 'rb') as f:
        sequence_data = pickle.load(f)
    return sequence_data['sequences']


def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {filename}")
        return 0, None


def train_model(model, dataloader, num_epochs, device, checkpoint_dir='checkpoints', resume=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start_epoch = 0
    if resume:
        latest_checkpoint = max([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')],
                                key=os.path.getctime, default=None)
        if latest_checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
            start_epoch += 1  # Start from the next epoch

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (sequences, masks) in enumerate(dataloader):
            sequences, masks = sequences.to(device), masks.to(device)
            print(f"Batch {batch_idx}:")
            print(f"  Sequences shape: {sequences.shape}")
            print(f"  Masks shape: {masks.shape}")

            optimizer.zero_grad()
            reconstructed, _ = model(sequences, masks)
            print(f"  Reconstructed shape: {reconstructed.shape}")

            loss = criterion(reconstructed * masks.unsqueeze(1), sequences.transpose(1, 2) * masks.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx == 0:  # 첫 번째 배치에 대해서만 상세 정보 출력
                print(f"  Sequences min: {sequences.min().item()}, max: {sequences.max().item()}")
                print(f"  Masks min: {masks.min().item()}, max: {masks.max().item()}")
                print(f"  Reconstructed min: {reconstructed.min().item()}, max: {reconstructed.max().item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

    print("Training finished.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sequences = load_sequences("AllSequences.pkl")
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    max_points = dataset.max_length
    model = PointNet2AutoEncoder(feature_dim=128, num_points=max_points).to(device)

    num_epochs = 100
    resume_training = False

    train_model(model, dataloader, num_epochs, device, resume=resume_training)

    model.eval()
    features = []
    with torch.no_grad():
        for sequences, masks in dataloader:
            sequences, masks = sequences.to(device), masks.to(device)
            _, batch_features = model(sequences, masks)
            features.append(batch_features.cpu().numpy())
    features = np.concatenate(features, axis=0)

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(features)

    print("Clustering finished.")
    print(f"Number of sequences in each cluster: {np.bincount(cluster_labels)}")

    torch.save(model.state_dict(), "pointnet2_autoencoder_final.pth")
    print("Final model saved.")