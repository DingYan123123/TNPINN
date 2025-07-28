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
hidden_size = 64
output_size = forecast_steps
HPM_hidden_size = 2
epochs = 1000
learningrate = 0.005
dropout_rate = 0.2
mc_samples = 100
batch_size = 32

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

# 小波变换去噪函数（软阈值法）
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

# 数据归一化
def min_max_normalization_per_feature(data):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    min_vals = torch.min(data, dim=0)[0]
    max_vals = torch.max(data, dim=0)[0]
    return (data - min_vals) / (max_vals - min_vals + 1e-6)

def custom_normalization(data):
    capacity_col = data[:, -1]
    min_capacity = np.min(capacity_col)
    max_capacity = np.max(capacity_col)
    capacity_col_normalized = (capacity_col - min_capacity) / (max_capacity - min_capacity + 1e-6)
    other_cols = data[:, :-1]
    min_vals = np.min(other_cols, axis=0)
    max_vals = np.max(other_cols, axis=0)
    other_cols_normalized = 2 * (other_cols - min_vals) / (max_vals - min_vals + 1e-6) - 1
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

# LSTM模型
class WindowedLSTM(nn.Module):
    def __init__(self, input_size, window_size, hidden_size, HPM_hidden_size, out_size, dropout_rate=dropout_rate):
        super(WindowedLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size * window_size, hidden_size)
        self.lstm0 = nn.LSTMCell(input_size, hidden_size)
        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hpm = LSTMHPM(out_size + hidden_size * 2 + input_size * window_size, HPM_hidden_size, input_size * window_size)

    def forward(self, input, mc_dropout=False):
        seq_len, batch_size, _ = input.size()
        outputs = []
        out_xs = []
        old_hts = []
        outputs_hiddens = []
        h_t0 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=input.device)
        c_t0 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=input.device)
        h_t1 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=input.device)
        c_t1 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=input.device)
        for i in range(seq_len):
            old_h = h_t0
            old_hts.append(old_h)
            window_data = input[i]
            h_t0, c_t0 = self.lstm_cell(window_data, (old_h, c_t0))
            if mc_dropout:
                h_t0 = self.dropout(h_t0)
                c_t0 = self.dropout(c_t0)
            h_t1, c_t1 = self.lstm1(h_t0, (h_t1, c_t1))
            if mc_dropout:
                h_t1 = self.dropout(h_t1)
                c_t1 = self.dropout(c_t1)
            output = self.linear(h_t1)
            # 在验证阶段，禁用梯度计算以避免 RuntimeError
            if not mc_dropout:
                output_hidden = torch.zeros(batch_size, hidden_size, device=input.device)
                output_x = torch.zeros(batch_size, input_size * window_size, device=input.device)
            else:
                output_hidden = torch.autograd.grad(
                    output, old_h,
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )[0] if old_h.requires_grad else torch.zeros(batch_size, hidden_size, device=input.device)
                output_x = torch.autograd.grad(
                    output, window_data,
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )[0] if window_data.requires_grad else torch.zeros(batch_size, input_size * window_size, device=input.device)
            if i == 0:
                output_hidden = torch.zeros(batch_size, hidden_size, device=input.device)
                outputs_hiddens.append(output_hidden)
                output_x = torch.zeros(batch_size, input_size * window_size, device=input.device)
                out_xs.append(output_x)
            else:
                outputs_hiddens.append(output_hidden)
                out_xs.append(output_x)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1).squeeze(2)
        out_xs = torch.cat(out_xs)
        outputs_hiddens = torch.cat(outputs_hiddens)
        outputs_hiddens = min_max_normalization_per_feature(outputs_hiddens)
        outputs = torch.transpose(outputs, 0, 1)
        old_hts = torch.cat(old_hts)
        old_hts = min_max_normalization_per_feature(old_hts)
        input_flat = torch.squeeze(input, 1)
        outputs = outputs.squeeze(1)
        output_data = torch.cat((outputs, outputs_hiddens, input_flat, old_hts), dim=1)
        G = self.hpm(output_data)
        F = out_xs - G
        AllF_x = torch.autograd.grad(
            F, output_data,
            grad_outputs=torch.ones_like(F),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return outputs, F, AllF_x

class LSTMHPM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMHPM, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x

# 蒙特卡洛不确定性估计
def mc_uncertainty_estimation(model, input_data, mc_samples=100):
    model.train()  # 保持Dropout活跃
    predictions = []
    for _ in range(mc_samples):
        pred, _, _ = model(input_data, mc_dropout=True)
        predictions.append(pred.detach().cpu().numpy())
    predictions = np.stack(predictions, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    return mean_pred, std_pred, ci_lower, ci_upper

# NEW: 不确定性指标
def calculate_uncertainty_metrics(test_np, mean_pred, std_pred, ci_lower, ci_upper, confidence=0.95):
    nll = 0.5 * np.log(2 * np.pi * (std_pred ** 2 + 1e-6)) + ((test_np - mean_pred) ** 2) / (2 * (std_pred ** 2 + 1e-6))
    nll_mean = np.mean(nll)
    coverage = np.mean((test_np >= ci_lower) & (test_np <= ci_upper))
    calibration_error = np.abs(coverage - confidence)
    mpiw = np.mean(ci_upper - ci_lower)
    return nll_mean, calibration_error, mpiw, coverage

# 早停类
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

# 对每个电池的数据进行处理
# 对每个电池的数据进行处理
for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    # print(file_name)
    # 提取后8列特征
    features = df.iloc[:, 1:].values  # 获取后8列特征
    last_column = df.iloc[:, -1].values  # 获取最后一列特征
    # last_column = np.apply_along_axis(wavelet_denoising, 0, last_column)
    # 应用小波变换去噪（软阈值处理）
    if last_column.shape[0] == features.shape[0]:
        # print(features.shape, last_column.shape)
        features = np.column_stack((features[:, 0:5], last_column / 1.1))
    else:
        features = np.column_stack((features[:, 0:5], last_column[:-1] / 1.1))

    # # 应用小波变换去噪（软阈值处理）
    features = np.apply_along_axis(wavelet_denoising, 0, features)  # 对每一列特征应用小波去噪
    # features = np.column_stack((features1, last_column))

    features = min_max_normalization_per_feature(features)
    # print(features[:, -1])
    # 创建数据和目标
    data, target = build_sequences(features, window_size, forecast_steps)

    # 将每个电池的数据保存到字典中
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses = [], []
nlls, cals, mpiws = [], [] ,[] # NEW
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

    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    setup_seed(0)
    model = WindowedLSTM(input_size, window_size, hidden_size, HPM_hidden_size, output_size, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

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
            Fx_cpu = Fx[1].cpu()
            Fxs_cpu = Fxs[1].cpu()
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

    # 评估
    test_np = test_target_tensor.cpu().numpy()
    mae, mse, rmse = evaluation(test_np, mean_pred)
    maes.append(mae * 100)
    rmses.append(rmse * 100)

    # NEW: 计算 NLL, Calibration Error, MPIW
    nll, calibration_error, mpiw, coverage = calculate_uncertainty_metrics(test_np, mean_pred, std_pred, ci_lower, ci_upper)
    nlls.append(nll)
    cals.append(calibration_error)
    mpiws.append(mpiw)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")
    print(f"NLL: {nll:.4f}, Calibration Error: {calibration_error:.4f}, MPIW: {mpiw:.4f}, Coverage: {coverage * 100:.2f}%")

    # 可视化部分保持不变
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"Training Loss over Epochs for {test_battery}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 1
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
print(f"Average NLL: {np.mean(nlls):.4f}")      # NEW
print(f"Average Calibration Error: {np.mean(cals):.4f}")  # NEW
print(f"Average MPIW: {np.mean(mpiws):.4f}")    # NEW
