import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class VAE(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 인코더의 출력 크기 계산
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, *input_shape)
            sample_output = self.encoder(sample_input)
            self.encoder_output_shape = sample_output.shape[1:]
            self.encoder_output_dim = np.prod(self.encoder_output_shape)

        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 출력을 0과 1 사이로 제한
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.encoder_output_shape)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 데이터 로드 및 전처리 함수 수정
def load_and_preprocess_data(file_path):
    with open(file_path, 'rb') as f:
        voxelized_sequences = pickle.load(f)

    # 리스트를 3D numpy 배열로 변환
    data = np.array(voxelized_sequences)

    # numpy 배열을 PyTorch 텐서로 변환
    data = torch.FloatTensor(data)
    data = data.unsqueeze(1)  # 채널 차원 추가 (N, 1, D, H, W)
    return data


def train_vae(model, data_loader, num_epochs, device):
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                print(f'Recon range: {recon_batch.min().item():.4f} - {recon_batch.max().item():.4f}')

        avg_loss = train_loss / len(data_loader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')


# 인코딩 추출
def extract_encodings(model, data_loader, device):
    model.eval()
    encodings = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            encodings.append(mu.cpu().numpy())
    return np.concatenate(encodings)


# 유사한 시퀀스 찾기
def find_similar_sequences(encodings, num_similar=5):
    similarity_matrix = cosine_similarity(encodings)
    np.fill_diagonal(similarity_matrix, -1)  # 자기 자신 제외

    similar_sequences = []
    for i in range(len(encodings)):
        similar_indices = similarity_matrix[i].argsort()[::-1][:num_similar]
        similar_sequences.append(similar_indices)

    return similar_sequences


# 데이터 로드 및 전처리 함수 수정


def load_and_preprocess_data(file_path):
    with open(file_path, 'rb') as f:
        voxelized_sequences = pickle.load(f)

    print("Loaded data type:", type(voxelized_sequences))
    print("Loaded data shape:", np.shape(voxelized_sequences))

    if isinstance(voxelized_sequences, list):
        # 리스트를 3D numpy 배열로 변환
        data = np.array(voxelized_sequences)
    else:
        data = voxelized_sequences

    print("Converted data shape:", data.shape)

    # numpy 배열을 PyTorch 텐서로 변환
    data = torch.FloatTensor(data)
    if len(data.shape) == 4:
        data = data.unsqueeze(1)  # 채널 차원 추가 (N, 1, D, H, W)

    print("Final data shape:", data.shape)
    return data


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

def collate_fn(batch):
    return torch.stack(batch)



def train_vae(model, data_loader, num_epochs, device, checkpoint_dir='checkpoints', resume=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    optimizer = optim.Adam(model.parameters())
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
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):
            if isinstance(data, list):
                data = data[0]  # 리스트인 경우 첫 번째 요소를 사용
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                print(f'Recon range: {recon_batch.min().item():.4f} - {recon_batch.max().item():.4f}')

        avg_loss = train_loss / len(data_loader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

        # 매 5 에폭마다 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)


# 메인 실행 코드 수정
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 데이터 로드 및 전처리
    data = load_and_preprocess_data("VoxelizedSequences.pkl")
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # VAE 모델의 입력 크기 설정
    input_shape = data.shape[2:]  # (depth, height, width)
    print("Input shape:", input_shape)

    # VAE 모델 초기화 및 학습
    latent_dim = 64
    model = VAE(latent_dim, input_shape).to(device)
    num_epochs = 10

    # 학습 재개 여부 설정 (True로 설정하면 가장 최근의 체크포인트에서 학습 재개)
    resume_training = False

    train_vae(model, data_loader, num_epochs, device, resume=resume_training)
    # 인코딩 추출
    encodings = extract_encodings(model, data_loader, device)

    # 유사한 시퀀스 찾기
    similar_sequences = find_similar_sequences(encodings)

    # 결과 출력 (예시)
    for i, similar in enumerate(similar_sequences[:5]):  # 처음 5개 시퀀스에 대해서만 출력
        print(f"Sequence {i}: Similar sequences are {similar}")