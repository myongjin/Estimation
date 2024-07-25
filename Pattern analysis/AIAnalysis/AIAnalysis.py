import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from autoencoder_model import Conv3DVAE
import os
import sys

# 상위 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import PattenAnalysisTool as pat

# GPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 사용 중인 디바이스 출력

# Load sequences
file_name = "AllSequences"
try:
    loaded_sequence_data = pat.load_sequences(file_name + '.pkl')
except FileNotFoundError:
    print(f"File {file_name}.pkl not found.")
    sys.exit(1)

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

# 데이터를 GPU로 이동
sequences_tensor = sequences_tensor.to(device)

batch_size = 32
dataset = TensorDataset(sequences_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# set model name
model_name = 'conv3dvae_model'
model_path = f"{model_name}.pth"


# Conv3DVAE 모델 설정
input_channels = 2
hidden_dims = [32, 64, 128]
latent_dim = 64

if os.path.exists(model_path):
    # 기존 모델 불러오기
    checkpoint = torch.load(model_path)
    model = Conv3DVAE(input_channels, hidden_dims, latent_dim, max_length).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"기존 모델을 불러왔습니다: {model_path}")
else:
    # 새로운 모델 생성
    model = Conv3DVAE(input_channels, hidden_dims, latent_dim, max_length).to(device)
    print(f"새로운 모델을 생성하였습니다: {model_path}")

optimizer = optim.Adam(model.parameters(), lr=0.001)


# 손실 함수 정의
def loss_function(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss


# 학습 관련 변수 설정
num_epochs = 100
print_every = 10  # 몇 에폭마다 출력할지 설정

# 학습 시작
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        batch = batch[0]  # 배치에서 시퀀스 데이터 추출
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(batch)
        loss = loss_function(recon_batch, batch, mu, log_var)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 에폭마다 평균 손실 계산
    epoch_loss /= len(dataloader)

    # print_every 에폭마다 출력
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, model_name)
print(f"학습된 모델이 저장되었습니다: {model_path}")
