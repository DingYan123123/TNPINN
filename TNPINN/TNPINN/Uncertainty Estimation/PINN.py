import os
from math import sqrt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

# 参数设置
window_size = 30
forecast_steps = 10
input_size = 180
hidden_size = 64
HPM_hidden_size = 4
output_size = forecast_steps
epochs = 1000
weight_decay = 0
learningrate = 0.001
batch_size = 32
dropout_rate = 0.1
mc_samples = 100

# 设置随机种子
def setup_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# 线性模型（添加 Dropout）
class Linear(nn.Module):
    def __init__(self, input_size, hidden_size, HPM_hidden_size, output_size, dropout_rate=dropout_rate):
        super(Linear, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.hpm = DeepHPM(output_size + input_size, HPM_hidden_size, input_size)

    def forward(self, input, mc_dropout=False):
        batch_size = input.size(0)
        input = input.view(batch_size, -1)  # 展平输入为 [batch_size, input_size]
        input.requires_grad_(True)
        hidden = self.linear1(input)
        hidden = self.tanh(hidden)
        if mc_dropout:
            hidden = self.dropout(hidden)
        output = self.linear2(hidden)  # 形状为 [batch_size, output_size]
        output_input = torch.autograd.grad(
            output, input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0] if input.requires_grad else torch.zeros_like(input)
        HPM_input = torch.cat((input, output), dim=1)  # 形状为 [batch_size, input_size + output_size]
        G = self.hpm(HPM_input)
        F = output_input - G
        HPM_input.requires_grad_(True)
        F_out = torch.autograd.grad(
            F, input,
            grad_outputs=torch.ones_like(F),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0] if input.requires_grad else torch.zeros_like(input)
        return output, F, F_out

class DeepHPM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepHPM, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x

# 评估函数
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    return mae, mse, rmse

# 归一化函数
def min_max_normalization_per_feature(data):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    min_vals = torch.min(data, dim=0)[0]
    max_vals = torch.max(data, dim=0)[0]
    return (data - min_vals) / (max_vals - min_vals + 1e-6)

# 小波去噪
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

# 生成时序序列
def build_sequences(text, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]
        target = text[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)

# 蒙特卡洛不确定性估计
def mc_uncertainty_estimation(model, input_data, mc_samples=100):
    model.train()  # 保持 Dropout 活跃
    predictions = []
    for _ in range(mc_samples):
        pred, _, _ = model(input_data, mc_dropout=True)
        predictions.append(pred.detach().cpu().numpy().squeeze())  # 确保移除多余维度
    predictions = np.stack(predictions, axis=0)  # 形状为 [mc_samples, n_samples, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)  # 形状为 [n_samples, forecast_steps]
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    return mean_pred, std_pred, ci_lower, ci_upper

# 早停机制
class EarlyStopping:
    def __init__(self, patience=100, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            return True
        return False

# 数据加载和处理
folder_path = '../NASA Cleaned'
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:].values
    last_column = df.iloc[:, -1].values
    last_column = np.apply_along_axis(wavelet_denoising, 0, last_column)
    features = np.column_stack((features[:, 0:5], last_column / 2))
    features = min_max_normalization_per_feature(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses = [], []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    train_data_tensor = torch.tensor(train_data, requires_grad=True, dtype=torch.float32).to(device)
    train_data_tensor = train_data_tensor.unsqueeze(1)
    train_data_tensor = train_data_tensor.reshape(len(train_data_tensor), 1, -1)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, requires_grad=True, dtype=torch.float32).to(device)
    test_data_tensor = test_data_tensor.unsqueeze(1)
    test_data_tensor = test_data_tensor.reshape(len(test_data_tensor), 1, -1)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    # 创建 DataLoader
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    setup_seed(0)
    model = Linear(input_size, hidden_size, HPM_hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=100, delta=0.01)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            output, Fx, Fxs = model(batch_data)
            output_cpu = output.cpu()
            batch_target_cpu = batch_target.cpu()
            Fx_cpu = Fx[1].cpu() if isinstance(Fx, tuple) else Fx.cpu()
            Fxs_cpu = Fxs[1].cpu() if isinstance(Fxs, tuple) else Fxs.cpu()
            loss = torch.sum((output_cpu.squeeze() - batch_target_cpu.squeeze()) ** 2) + torch.sum(Fx_cpu ** 2) + torch.sum(Fxs_cpu ** 2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        pred, _, _ = model(test_data_tensor, mc_dropout=False)
        pred_np = pred.detach().squeeze().cpu().numpy()
        test_np = test_target_tensor.detach().squeeze().cpu().numpy()
        val_loss = np.mean((pred_np - test_np) ** 2)
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 蒙特卡洛不确定性估计
    mean_pred, std_pred, ci_lower, ci_upper = mc_uncertainty_estimation(model, test_data_tensor, mc_samples)

    # 确保 mean_pred 和 test_np 形状一致
    mean_pred = mean_pred.squeeze()  # 移除多余维度
    test_np = test_np.squeeze()

    # 评估
    mae, mse, rmse = evaluation(test_np, mean_pred)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")
    maes.append(mae * 100)
    rmses.append(rmse * 100)

    # 计算覆盖概率
    coverage = np.mean((test_np >= ci_lower) & (test_np <= ci_upper))
    print(f"95% Coverage Probability: {coverage * 100:.2f}%")

    # 绘制训练损失图
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"Training Loss over Epochs for {test_battery}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制预测和置信区间（示例：第一个样本）
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 20
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

    # 绘制预测与真实值的散点图
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

# 汇总交叉验证结果
print("\nCross-validation results:")
print(f"Average RMSE: {np.mean(rmses):.3f}")
print(f"Average MAE: {np.mean(maes):.3f}")