import os
from math import sqrt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 参数设置
window_size = 30
forecast_steps = 10
input_size = 6
hidden_size = 32
output_size = forecast_steps
epochs = 1000
learningrate = 0.01
num_heads = 3
conv_channels = 4
kernel_size = 2
weight_decay = 1e-4
max_size = 2
batch_size = 32
patience = 50
dropout_rate = 0.2  # Dropout率，用于蒙特卡洛不确定性估计
mc_samples = 100  # 蒙特卡洛采样次数

# 设置随机种子保证复现性
def setup_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# AttentionCNNLSTM模型（加入Dropout）
class AttentionCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, conv_channels=conv_channels, kernel_size=kernel_size, num_heads=num_heads, dropout_rate=dropout_rate):
        super(AttentionCNNLSTM, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=max_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # 增加Dropout层
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x, mc_dropout=False):
        attn_output, _ = self.attn(x, x, x)
        x = attn_output
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        if mc_dropout:
            x = self.dropout(x)  # 在推理阶段保留Dropout
        lstm_out, _ = self.lstm(x)
        if mc_dropout:
            lstm_out = self.dropout(lstm_out)
        out = self.out(lstm_out[:, -1, :])
        return out

# 小波变换去噪函数（软阈值法）
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

# 数据归一化
def min_max_normalization_per_feature(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)  # 防止除零

def custom_normalization(data):
    capacity_col = data[:, -1]
    min_capacity = np.min(capacity_col)
    max_capacity = np.max(capacity_col)
    capacity_col_normalized = (capacity_col - min_capacity) / (max_capacity - min_capacity + 1e-8)
    other_cols = data[:, :-1]
    min_vals = np.min(other_cols, axis=0)
    max_vals = np.max(other_cols, axis=0)
    other_cols_normalized = 2 * (other_cols - min_vals) / (max_vals - min_vals + 1e-8) - 1
    return np.column_stack((other_cols_normalized, capacity_col_normalized))

# 构建序列样本
def build_sequences(text, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]
        target = text[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)

# 模型评估
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    return mae, mse, rmse

# 蒙特卡洛不确定性估计
def mc_uncertainty_estimation(model, input_data, mc_samples=100):
    model.train()  # 保持Dropout活跃
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            pred = model(input_data, mc_dropout=True)
        predictions.append(pred.cpu().numpy())
    predictions = np.stack(predictions, axis=0)  # [mc_samples, batch_size, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred  # 95%置信区间下界
    ci_upper = mean_pred + 1.96 * std_pred  # 95%置信区间上界
    return mean_pred, std_pred, ci_lower, ci_upper

# 加载数据
folder_path = '../NASA Cleaned'  # 替换为你的CSV文件夹路径
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

# 对每个电池的数据进行处理
for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:].values
    last_column = df.iloc[:, -1].values
    features = np.column_stack((features[:, 0:5], last_column / 2))
    features = min_max_normalization_per_feature(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)

# 训练与测试
maes, rmses = [], []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

for test_battery, (test_data, test_target) in battery_data.items():
    print(f"\nTesting on battery: {test_battery}")
    train_data, train_target = [], []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)
    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 参数配置
    setup_seed(0)
    model = AttentionCNNLSTM(input_size=input_size, hidden_size=hidden_size, out_size=output_size,
                             num_heads=num_heads, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learningrate)

    train_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_output = model(test_data_tensor)
            val_loss = criterion(val_output, test_target_tensor)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    # 蒙特卡洛不确定性估计
    mean_pred, std_pred, ci_lower, ci_upper = mc_uncertainty_estimation(model, test_data_tensor, mc_samples)

    # 评估
    test_np = test_target_tensor.cpu().numpy()
    mae, mse, rmse = evaluation(test_np, mean_pred)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")
    maes.append(mae * 100)
    rmses.append(rmse * 100)

    # 计算覆盖概率
    coverage = np.mean((test_np >= ci_lower) & (test_np <= ci_upper))
    print(f"95% Coverage Probability: {coverage * 100:.2f}%")

    # 可视化：训练损失
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"Training Loss over Epochs for {test_battery}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 可视化：预测和置信区间（示例：第一个样本的预测序列）
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 1  # 第一个样本
    plt.plot(time_steps, test_np[sample_idx], label="True", marker='o')
    plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x')
    plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                     color="red", alpha=0.2, label="95% CI")
    plt.title(f"Prediction with 95% CI for {test_battery}", fontsize=18)
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("SOH(%)", fontsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.show()

    # 可视化：散点图
    test_np_flat = test_np.flatten()
    pred_np_flat = mean_pred.flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(test_np_flat, pred_np_flat, s=100, color='dodgerblue', alpha=0.8)
    plt.plot([min(test_np_flat), max(test_np_flat)], [min(test_np_flat), max(test_np_flat)], 'r--', label='Ideal')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# 总体评估结果
print("\nCross-validation results:")
print(f"Average RMSE: {np.mean(rmses):.3f}")
print(f"Average MAE: {np.mean(maes):.3f}")