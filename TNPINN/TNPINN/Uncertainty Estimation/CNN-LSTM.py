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
import scipy.stats as stats

# 参数设置
window_size = 30
forecast_steps = 10
input_size = 6
hidden_size = 32
output_size = forecast_steps
conv_channels = 6
kernel_size = 2
pool_size = 2
epochs = 1000
weight_decay = 0
learningrate = 0.001
batch_size = 32
dropout_rate = 0.3  # Dropout率，用于蒙特卡洛不确定性估计
mc_samples = 100  # 蒙特卡洛采样次数


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


# 小波变换去噪函数
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


# CNN-LSTM模型（加入Dropout）
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, conv_channels=conv_channels, kernel_size=kernel_size,
                 pool_size=pool_size, dropout_rate=0.3):
        super(CNNLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout_rate)  # 增加Dropout层
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x, mc_dropout=False):
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        if mc_dropout:
            x = self.dropout(x)  # 在推理阶段保留Dropout
        output, _ = self.lstm(x)
        output = self.linear(output[:, -1, :])
        return output


# 数据归一化处理
def custom_normalization(data):
    capacity_col = data[:, -1]
    min_capacity = np.min(capacity_col)
    max_capacity = np.max(capacity_col)
    capacity_col_normalized = (capacity_col - min_capacity) / (max_capacity - min_capacity)
    other_cols = data[:, :-1]
    min_vals = np.min(other_cols, axis=0)
    max_vals = np.max(other_cols, axis=0)
    other_cols_normalized = 2 * (other_cols - min_vals) / (max_vals - min_vals) - 1
    return np.column_stack((other_cols_normalized, capacity_col_normalized))


# 生成时序序列
def build_sequences(text, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]
        target = text[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)


# 评估函数
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    return mae, mse, rmse


# 不确定性评估指标
def calculate_uncertainty_metrics(y_true, ci_lower, ci_upper, confidence_level=0.95, eta=None):
    """
    计算不确定性评估指标：PICP, PINAW, CWC

    Args:
        y_true: 真实值 [n_samples, forecast_steps]
        ci_lower: 置信区间下界 [n_samples, forecast_steps]
        ci_upper: 置信区间上界 [n_samples, forecast_steps]
        confidence_level: 置信水平，默认0.95
        eta: 惩罚因子，如果为None则自适应设置

    Returns:
        picp: Prediction Interval Coverage Probability
        pinaw: Prediction Interval Normalized Average Width
        cwc: Coverage Width-based Criterion
        eta_used: 实际使用的eta值
    """
    # PICP (Prediction Interval Coverage Probability)
    coverage = (y_true >= ci_lower) & (y_true <= ci_upper)
    picp = np.mean(coverage)

    # PINAW (Prediction Interval Normalized Average Width)
    interval_width = ci_upper - ci_lower
    y_range = np.max(y_true) - np.min(y_true)
    pinaw = np.mean(interval_width) / y_range

    # 自适应eta设置
    if eta is None:
        coverage_deficit = confidence_level - picp
        if coverage_deficit <= 0:
            eta = 0.1  # 覆盖率满足时使用较小的eta
        elif coverage_deficit <= 0.05:  # 覆盖率接近目标
            eta = 0.3
        elif coverage_deficit <= 0.10:  # 覆盖率中等偏差
            eta = 0.5
        else:  # 覆盖率严重不足
            eta = 1.0

    # CWC (Coverage Width-based Criterion)
    if picp >= confidence_level:
        # 如果覆盖率满足要求，CWC = PINAW
        cwc = pinaw
    else:
        # 如果覆盖率不足，增加惩罚项
        coverage_penalty = eta * (confidence_level - picp) / confidence_level
        cwc = pinaw * (1 + coverage_penalty)

    return picp, pinaw, cwc, eta


# 早停类
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0


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
    ci_lower = mean_pred - 1.96 * std_pred  # 95%置信区间
    ci_upper = mean_pred + 1.96 * std_pred
    return mean_pred, std_pred, ci_lower, ci_upper


# 数据加载
folder_path = '../NASA Cleaned'
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

for file_name in battery_files:
    df = pd.read_csv(os.path.join(folder_path, file_name))
    features = df.iloc[:, 1:].values
    # features = np.apply_along_axis(wavelet_denoising, 0, features)  # 可选：小波去噪
    features = custom_normalization(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses = [], []
picps, pinaws, cwcs, etas = [], [], [], []  # 添加不确定性指标存储
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

for test_battery, (test_data, test_target) in battery_data.items():
    print(f"Testing on battery: {test_battery}")

    # 训练集数据
    train_data = []
    train_target = []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)
    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    # 转换为Tensor
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    # 创建DataLoader
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    setup_seed(0)
    model = CNNLSTM(input_size=input_size, hidden_size=hidden_size, out_size=output_size,
                    conv_channels=conv_channels, kernel_size=kernel_size, pool_size=pool_size,
                    dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learningrate)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 训练
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(test_data_tensor)
            val_loss = criterion(val_pred, test_target_tensor)
            val_losses.append(val_loss.item())

        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    # 蒙特卡洛不确定性估计
    mean_pred, std_pred, ci_lower, ci_upper = mc_uncertainty_estimation(model, test_data_tensor, mc_samples)

    # 评估
    test_np = test_target_tensor.cpu().numpy()
    mae, mse, rmse = evaluation(test_np, mean_pred)
    print(f"RMSE: {rmse * 100:.3f}%, MAE: {mae * 100:.3f}%")

    # 计算不确定性指标
    picp, pinaw, cwc, eta_used = calculate_uncertainty_metrics(test_np, ci_lower, ci_upper)
    print(f"PICP (95% Coverage): {picp * 100:.2f}%")
    print(f"PINAW (Normalized Width): {pinaw:.4f}")
    print(f"CWC (Coverage-Width Criterion): {cwc:.4f} (eta={eta_used:.1f})")

    maes.append(mae * 100)
    rmses.append(rmse * 100)
    picps.append(picp)
    pinaws.append(pinaw)
    cwcs.append(cwc)
    etas.append(eta_used)

    # 绘制训练损失图
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title(f"Training & Validation Loss for {test_battery}", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 绘制预测与真实值的对比图
    test_np_flat = test_np.flatten()
    mean_pred_flat = mean_pred.flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(test_np_flat, mean_pred_flat, s=100, color='dodgerblue', alpha=0.8)
    plt.plot([min(test_np_flat), max(test_np_flat)], [min(test_np_flat), max(test_np_flat)],
             'r--', label='Ideal', linewidth=2)
    plt.xlabel("True Values", fontsize=14)
    plt.ylabel("Predicted Values", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.title(f"Prediction vs True Values for {test_battery}", fontsize=14)
    plt.show()

    # 绘制置信区间图（放大字体）
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = min(30, len(test_np) - 1)  # 防止索引越界
    plt.plot(time_steps, test_np[sample_idx], label="True", marker='o', linewidth=2, markersize=8)
    plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x', linewidth=2, markersize=8)
    plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                     color="red", alpha=0.2, label="95% CI")
    plt.title(f"Prediction with 95% CI for {test_battery}", fontsize=16)
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("SOH", fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()

    # 绘制不确定性分析图
    plt.figure(figsize=(15, 5))

    # 子图1：覆盖率分布
    plt.subplot(1, 3, 1)
    coverage_per_sample = np.mean((test_np >= ci_lower) & (test_np <= ci_upper), axis=1)
    plt.hist(coverage_per_sample, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.95, color='red', linestyle='--', linewidth=2, label='Target 95%')
    plt.axvline(x=np.mean(coverage_per_sample), color='green', linestyle='-', linewidth=2,
                label=f'Actual {np.mean(coverage_per_sample):.2f}')
    plt.xlabel('Coverage Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Coverage Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：区间宽度分布
    plt.subplot(1, 3, 2)
    interval_widths = ci_upper - ci_lower
    plt.hist(interval_widths.flatten(), bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Interval Width', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Interval Width Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 子图3：不确定性指标对比
    plt.subplot(1, 3, 3)
    metrics = ['PICP', 'PINAW', 'CWC']
    values = [picp, pinaw, cwc]
    colors = ['lightgreen', 'orange', 'purple']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Uncertainty Metrics', fontsize=14)
    # 在柱状图上显示数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("-" * 60)

# 汇总结果
print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS SUMMARY (CNN-LSTM)")
print("=" * 80)
print(f"Average RMSE: {np.mean(rmses):.3f}% ± {np.std(rmses):.3f}%")
print(f"Average MAE: {np.mean(maes):.3f}% ± {np.std(maes):.3f}%")
print("\nUNCERTAINTY QUANTIFICATION METRICS:")
print(f"Average PICP: {np.mean(picps) * 100:.2f}% ± {np.std(picps) * 100:.2f}%")
print(f"Average PINAW: {np.mean(pinaws):.4f} ± {np.std(pinaws):.4f}")
print(f"Average CWC: {np.mean(cwcs):.4f} ± {np.std(cwcs):.4f} (avg eta={np.mean(etas):.1f})")
print("=" * 80)

# 详细结果表格
print("\nDETAILED RESULTS BY BATTERY:")
print("-" * 120)
print(f"{'Battery':<20} {'RMSE(%)':<10} {'MAE(%)':<10} {'PICP(%)':<10} {'PINAW':<8} {'CWC':<8} {'ETA':<5}")
print("-" * 120)
for i, (battery_name, _) in enumerate(battery_data.items()):
    print(f"{battery_name:<20} {rmses[i]:<10.3f} {maes[i]:<10.3f} "
          f"{picps[i] * 100:<10.2f} {pinaws[i]:<8.4f} {cwcs[i]:<8.4f} {etas[i]:<5.1f}")
print("-" * 120)

# 绘制综合结果对比图
plt.figure(figsize=(15, 10))

# 子图1：RMSE和MAE对比
plt.subplot(2, 3, 1)
battery_names = [name.replace('.csv', '') for name, _ in battery_data.items()]
x_pos = np.arange(len(battery_names))
width = 0.35
plt.bar(x_pos - width / 2, rmses, width, label='RMSE', alpha=0.8, color='skyblue')
plt.bar(x_pos + width / 2, maes, width, label='MAE', alpha=0.8, color='lightcoral')
plt.xlabel('Battery')
plt.ylabel('Error (%)')
plt.title('RMSE vs MAE by Battery')
plt.xticks(x_pos, battery_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：PICP分布
plt.subplot(2, 3, 2)
plt.bar(x_pos, [p * 100 for p in picps], alpha=0.8, color='lightgreen')
plt.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target 95%')
plt.xlabel('Battery')
plt.ylabel('PICP (%)')
plt.title('Coverage Probability by Battery')
plt.xticks(x_pos, battery_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图3：PINAW分布
plt.subplot(2, 3, 3)
plt.bar(x_pos, pinaws, alpha=0.8, color='orange')
plt.xlabel('Battery')
plt.ylabel('PINAW')
plt.title('Normalized Width by Battery')
plt.xticks(x_pos, battery_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# 子图4：CWC分布
plt.subplot(2, 3, 4)
plt.bar(x_pos, cwcs, alpha=0.8, color='purple')
plt.xlabel('Battery')
plt.ylabel('CWC')
plt.title('Coverage-Width Criterion by Battery')
plt.xticks(x_pos, battery_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# 子图5：ETA分布
plt.subplot(2, 3, 5)
plt.bar(x_pos, etas, alpha=0.8, color='brown')
plt.xlabel('Battery')
plt.ylabel('ETA')
plt.title('Adaptive ETA by Battery')
plt.xticks(x_pos, battery_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# 子图6：综合性能雷达图（使用前5个电池）
plt.subplot(2, 3, 6)
categories = ['RMSE\n(Lower Better)', 'MAE\n(Lower Better)', 'PICP\n(Higher Better)',
              'PINAW\n(Lower Better)', 'CWC\n(Lower Better)']

# 标准化指标到0-1范围用于雷达图
rmse_norm = 1 - np.array(rmses[:5]) / max(rmses)  # 反转，越小越好
mae_norm = 1 - np.array(maes[:5]) / max(maes)  # 反转，越小越好
picp_norm = np.array(picps[:5])  # 越大越好
pinaw_norm = 1 - np.array(pinaws[:5]) / max(pinaws)  # 反转，越小越好
cwc_norm = 1 - np.array(cwcs[:5]) / max(cwcs)  # 反转，越小越好

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

for i in range(min(5, len(battery_names))):
    values = [rmse_norm[i], mae_norm[i], picp_norm[i], pinaw_norm[i], cwc_norm[i]]
    values += values[:1]  # 闭合雷达图
    plt.plot(angles, values, 'o-', linewidth=2, label=battery_names[i][:10])

plt.xticks(angles[:-1], categories, fontsize=8)
plt.ylim(0, 1)
plt.title('Performance Radar Chart\n(Normalized Metrics)', fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nModel Architecture: CNN-LSTM with Monte Carlo Dropout")
print(f"Total Parameters: Conv1D + LSTM + Linear layers")
print(f"Uncertainty Method: Monte Carlo Dropout ({mc_samples} samples)")
print(f"Confidence Level: 95%")
print("=" * 80)