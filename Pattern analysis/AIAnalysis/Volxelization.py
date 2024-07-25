import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_sequences(file_name):
    with open(file_name, 'rb') as f:
        sequence_data = pickle.load(f)
    return sequence_data


def center_sequences(sequences):
    centered_sequences = []
    for seq in sequences:
        center = np.mean(seq, axis=0)
        centered_seq = seq - center
        centered_sequences.append(centered_seq)
    return centered_sequences


def create_bounding_box(sequences):
    min_coords = np.min([np.min(seq, axis=0) for seq in sequences], axis=0)
    max_coords = np.max([np.max(seq, axis=0) for seq in sequences], axis=0)
    return min_coords, max_coords


def visualize_voxel(voxel, title='Voxelized Sequence'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel, edgecolor='k')
    plt.title(title)
    plt.show()


def voxelize_sequences(sequences, min_coords, max_coords, voxel_size):
    # 전체 범위 계산
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    range_z = max_coords[2] - min_coords[2]

    # 각 축의 분할 개수 계산 (올림하여 정수로)
    div_x = int(np.ceil(range_x / voxel_size))
    div_y = int(np.ceil(range_y / voxel_size))
    div_z = int(np.ceil(range_z / voxel_size))

    voxelized_sequences = []

    for seq in sequences:
        voxels = np.zeros((div_x, div_y, div_z), dtype=int)
        for i in range(len(seq) - 1):
            start = seq[i]
            end = seq[i + 1]

            # 선분을 따라 여러 점을 샘플링
            num_samples = max(int(np.ceil(np.linalg.norm(end - start) / (voxel_size / 2))), 2)
            samples = np.linspace(start, end, num_samples)

            for point in samples:
                x_idx = min(int((point[0] - min_coords[0]) / voxel_size), div_x - 1)
                y_idx = min(int((point[1] - min_coords[1]) / voxel_size), div_y - 1)
                z_idx = min(int((point[2] - min_coords[2]) / voxel_size), div_z - 1)
                voxels[x_idx, y_idx, z_idx] = 1

        voxelized_sequences.append(voxels)

    return voxelized_sequences


def visualize_sequence_and_voxel(sequence, voxel, min_coords, voxel_size, title='Sequence and Voxel'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 원래 시퀀스 플롯
    ax.plot(sequence[:, 0], sequence[:, 1], sequence[:, 2], label='Original Sequence', color='blue', linewidth=2)

    # 복셀화된 시퀀스 플롯
    voxel_shape = voxel.shape

    # 데이터가 존재하는 복셀의 범위 찾기
    non_zero = np.nonzero(voxel)
    if len(non_zero[0]) > 0:  # 데이터가 존재하는 경우
        x_min, y_min, z_min = [np.min(nz) for nz in non_zero]
        x_max, y_max, z_max = [np.max(nz) for nz in non_zero]
    else:  # 데이터가 존재하지 않는 경우 (예외 처리)
        x_min, y_min, z_min = 0, 0, 0
        x_max, y_max, z_max = voxel_shape[0] - 1, voxel_shape[1] - 1, voxel_shape[2] - 1

    # 복셀 그리기
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            for z in range(z_min, z_max + 1):
                if voxel[x, y, z]:
                    x_pos = min_coords[0] + x * voxel_size
                    y_pos = min_coords[1] + y * voxel_size
                    z_pos = min_coords[2] + z * voxel_size
                    ax.bar3d(x_pos, y_pos, z_pos, voxel_size, voxel_size, voxel_size,
                             color='red', alpha=0.3, edgecolor='k')

    # 범례를 위한 가짜 플롯
    ax.plot([], [], [], color='red', alpha=0.3, label='Voxelized Sequence')

    # 축 범위 설정 (데이터가 존재하는 부분만)
    margin = voxel_size  # 여백 추가
    ax.set_xlim(min_coords[0] + x_min * voxel_size - margin,
                min_coords[0] + (x_max + 1) * voxel_size + margin)
    ax.set_ylim(min_coords[1] + y_min * voxel_size - margin,
                min_coords[1] + (y_max + 1) * voxel_size + margin)
    ax.set_zlim(min_coords[2] + z_min * voxel_size - margin,
                min_coords[2] + (z_max + 1) * voxel_size + margin)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    # 정육면체 형태로 보이도록 aspect ratio 설정
    ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    plt.show()
# Load sequences
file_name = "ID002_Sequence.pkl"
sequence_data = load_sequences(file_name)

# Center sequences to the origin
sequences = sequence_data['sequences']
centered_sequences = center_sequences(sequences)
min_coords, max_coords = create_bounding_box(centered_sequences)

sampleIdx=100
voxel_size = 10  # 복셀의 크기 설정
voxelized_sequences = voxelize_sequences(centered_sequences, min_coords, max_coords, voxel_size)

visualize_sequence_and_voxel(centered_sequences[sampleIdx], voxelized_sequences[sampleIdx], min_coords, voxel_size,
                             title='Sequence and Voxel Example')


# Save voxelized sequences to a file
voxel_file_name = "ID002_Sequence_Voxel.pkl"
with open(voxel_file_name, 'wb') as f:
    pickle.dump(voxelized_sequences, f)
