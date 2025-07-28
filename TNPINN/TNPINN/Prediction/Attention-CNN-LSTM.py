import os
from math import sqrt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from torch.nn.functional import dropout
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from scipy.stats import gaussian_kde

# 参数设置
window_size = 20
forecast_steps = 10
input_size = 6
hidden_size = 128
output_size = forecast_steps
epochs = 1000
learningrate = 0.001
num_heads = 3
conv_channels = 4
kernel_size = 2
weight_decay = 1e-4
max_size = 2
batch_size = 32
patience = 50
dropout_rate = 0.2  # Dropout rate for MC uncertainty
mc_samples = 100  # Monte Carlo samples
alpha = 0.05  # Significance level for 95% CI
rho = 2  # CWC hyperparameter

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

# 数据归一化
def min_max_normalization_per_feature(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-6)  # Add epsilon to avoid division by zero

# 归一化处理
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

# 不确定性量化指标计算
def calculate_uncertainty_metrics(y_true, ci_lower, ci_upper, alpha=0.05):
    """
    计算PICP, PINMW, CWC指标

    Args:
        y_true: 真实值 [N, forecast_steps]
        ci_lower: 置信区间下界 [N, forecast_steps]
        ci_upper: 置信区间上界 [N, forecast_steps]
        alpha: 显著性水平

    Returns:
        dict: 包含PICP, PINMW, CWC的字典
    """
    coverage_mask = (y_true >= ci_lower) & (y_true <= ci_upper)
    picp = np.mean(coverage_mask)
    interval_width = ci_upper - ci_lower
    y_range = np.max(y_true) - np.min(y_true)
    pinmw = np.mean(interval_width) / y_range if y_range != 0 else 0
    eta = 1 if picp < (1 - alpha) else 0
    cwc = pinmw * (1 + eta * np.exp(-rho * (picp - (1 - alpha))))
    return {
        'PICP': picp,
        'PINMW': pinmw,
        'CWC': cwc,
        'Coverage_Rate': picp * 100
    }

# 蒙特卡洛不确定性估计
def mc_uncertainty_estimation(model, input_data, mc_samples=100):
    model.train()  # Keep Dropout active for MC
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            pred = model(input_data, mc_dropout=True)
        predictions.append(pred.cpu().numpy())
    predictions = np.stack(predictions, axis=0)  # [mc_samples, batch_size, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred  # 95% CI
    ci_upper = mean_pred + 1.96 * std_pred
    return mean_pred, std_pred, ci_lower, ci_upper, predictions

# 绘制概率密度图（学术论文风格）
def plot_probability_density(predictions_array, y_true, sample_idx=0, step_idx=0, battery_name="Battery"):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(6, 4))
    pred_dist = predictions_array[:, sample_idx, step_idx]
    true_value = y_true[sample_idx, step_idx]
    kde = gaussian_kde(pred_dist)
    x_min, x_max = pred_dist.min(), pred_dist.max()
    x_range = x_max - x_min
    x_smooth = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)
    density = kde(x_smooth)
    colors = ['blue', 'red', 'green', 'magenta', 'orange', 'purple', 'brown', 'pink']
    color = colors[sample_idx % len(colors)]
    ax.plot(x_smooth, density, color=color, linewidth=3, alpha=0.8)
    ax.fill_between(x_smooth, density, alpha=0.3, color=color)
    ax.axvline(true_value, color='black', linestyle='--', linewidth=2)
    max_density = np.max(density)
    ax.annotate('Actual value',
                xy=(true_value, max_density * 0.7),
                xytext=(true_value + 0.05, max_density * 0.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=14, ha='left')
    ax.set_xlabel('Capacity (AH)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability density', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# 绘制多个电池的概率密度图（2x2网格）
def plot_multiple_probability_densities(all_predictions, all_y_true, battery_names, sample_idx=0, step_idx=0):
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ['blue', 'red', 'green', 'magenta']
    for i in range(min(4, len(all_predictions))):
        ax = axes[i]
        predictions_array = all_predictions[i]
        y_true = all_y_true[i]
        if sample_idx < predictions_array.shape[1] and step_idx < predictions_array.shape[2]:
            pred_dist = predictions_array[:, sample_idx, step_idx]
            true_value = y_true[sample_idx, step_idx]
            kde = gaussian_kde(pred_dist)
            x_min, x_max = pred_dist.min(), pred_dist.max()
            x_range = x_max - x_min
            x_smooth = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)
            density = kde(x_smooth)
            ax.plot(x_smooth, density, color=colors[i], linewidth=3)
            ax.fill_between(x_smooth, density, alpha=0.3, color=colors[i])
            ax.axvline(true_value, color='black', linestyle='--', linewidth=2)
            max_density = np.max(density)
            ax.annotate('Actual value',
                        xy=(true_value, max_density * 0.6),
                        xytext=(true_value + 0.02, max_density * 0.8),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                        fontsize=12, ha='left')
            ax.set_title(f'Battery #{i + 5}', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Capacity (AH)', fontsize=12)
            ax.set_ylabel('Probability density', fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.set_ylim(bottom=0)
    fig.suptitle('Probability density of the predicted points at the 100th cycle (Attention-CNN-LSTM)',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# 绘制不确定性指标汇总图
def plot_uncertainty_metrics_summary(all_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics_names = ['PICP', 'PINMW', 'CWC', 'Coverage_Rate']
    titles = ['PICP (Prediction Interval Coverage Probability)',
              'PINMW (Prediction Interval Normalized Mean Width)',
              'CWC (Coverage Width-based Criterion)',
              'Coverage Rate (%)']
    for i, (metric, title) in enumerate(zip(metrics_names, titles)):
        ax = axes[i // 2, i % 2]
        values = [m[metric] for m in all_metrics]
        ax.bar(range(len(values)), values, alpha=0.7, color=f'C{i}')
        ax.set_xlabel('Battery Index', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        for j, v in enumerate(values):
            ax.text(j, v + max(values) * 0.01, f'{v:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

# Attention-CNN-LSTM模型
class AttentionCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, conv_channels=conv_channels, kernel_size=kernel_size, num_heads=num_heads, dropout_rate=0.2):
        super(AttentionCNNLSTM, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=max_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Add Dropout layer
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
            x = self.dropout(x)  # Apply Dropout during inference for MC
        lstm_out, _ = self.lstm(x)
        out = self.out(lstm_out[:, -1, :])
        return out

# 数据加载
folder_path = '../NASA Cleaned'
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:].values
    last_column = df.iloc[:, -1].values
    features = np.column_stack((features[:, 0:5], last_column / 2))
    features = min_max_normalization_per_feature(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses = [], []
all_uncertainty_metrics = []
all_predictions_for_density = []
all_y_true_for_density = []
battery_names_list = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

for battery_idx, (test_battery, (test_data, test_target)) in enumerate(battery_data.items()):
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

    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    # 预测和不确定性估计
    model.eval()
    with torch.no_grad():
        pred = model(test_data_tensor)
    pred_np = pred.detach().squeeze().cpu().numpy()
    test_np = test_target_tensor.detach().squeeze().cpu().numpy()

    # MC 不确定性估计
    mean_pred, std_pred, ci_lower, ci_upper, predictions_array = mc_uncertainty_estimation(model, test_data_tensor, mc_samples)

    # 存储数据用于概率密度图
    all_predictions_for_density.append(predictions_array)
    all_y_true_for_density.append(test_np)
    battery_names_list.append(test_battery)

    # 评估
    mae, mse, rmse = evaluation(test_np, mean_pred)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")
    maes.append(mae * 100)
    rmses.append(rmse * 100)

    # 不确定性指标
    uncertainty_metrics = calculate_uncertainty_metrics(test_np, ci_lower, ci_upper, alpha)
    all_uncertainty_metrics.append(uncertainty_metrics)
    print(f"PICP: {uncertainty_metrics['PICP']:.4f}")
    print(f"PINMW: {uncertainty_metrics['PINMW']:.4f}")
    print(f"CWC: {uncertainty_metrics['CWC']:.4f}")
    print(f"95% Coverage Rate: {uncertainty_metrics['Coverage_Rate']:.2f}%")

    # 绘制单个电池的概率密度图
    if battery_idx < 4:
        plot_probability_density(predictions_array, test_np,
                                sample_idx=0, step_idx=0,
                                battery_name=f"Battery #{battery_idx + 5} (Attention-CNN-LSTM)")

    # 绘制训练损失
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"Training Loss over Epochs for {test_battery}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制预测与真实值散点图
    test_np_flat = test_np.flatten()
    mean_pred_flat = mean_pred.flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(test_np_flat, mean_pred_flat, s=100, color='dodgerblue', alpha=0.8)
    plt.plot([min(test_np_flat), max(test_np_flat)], [min(test_np_flat), max(test_np_flat)], 'r--', label='Ideal')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制置信区间图
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 90
    plt.plot(time_steps, test_np[sample_idx], label="True", marker='o', linewidth=2)
    plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x', linewidth=2)
    plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                     color="red", alpha=0.2, label="95% CI")
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("SOH(%)", fontsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.show()

# 绘制多电池概率密度图
print("\nGenerating Figure 6 style probability density plots...")
plot_multiple_probability_densities(all_predictions_for_density[:4],
                                   all_y_true_for_density[:4],
                                   battery_names_list[:4],
                                   sample_idx=0, step_idx=0)

# 绘制不确定性指标汇总图
plot_uncertainty_metrics_summary(all_uncertainty_metrics)

# 总体评估结果
print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS SUMMARY (Attention-CNN-LSTM)")
print("=" * 60)
print(f"Average RMSE: {np.mean(rmses):.3f}%")
print(f"Average MAE: {np.mean(maes):.3f}%")

# 不确定性指标统计
avg_picp = np.mean([m['PICP'] for m in all_uncertainty_metrics])
avg_pinmw = np.mean([m['PINMW'] for m in all_uncertainty_metrics])
avg_cwc = np.mean([m['CWC'] for m in all_uncertainty_metrics])
avg_coverage = np.mean([m['Coverage_Rate'] for m in all_uncertainty_metrics])

print(f"\nUncertainty Quantification Metrics (Attention-CNN-LSTM):")
print(f"Average PICP: {avg_picp:.4f}")
print(f"Average PINMW: {avg_pinmw:.4f}")
print(f"Average CWC: {avg_cwc:.4f}")
print(f"Average Coverage Rate: {avg_coverage:.2f}%")

print(f"\nStandard Deviations:")
print(f"PICP Std: {np.std([m['PICP'] for m in all_uncertainty_metrics]):.4f}")
print(f"PINMW Std: {np.std([m['PINMW'] for m in all_uncertainty_metrics]):.4f}")
print(f"CWC Std: {np.std([m['CWC'] for m in all_uncertainty_metrics]):.4f}")