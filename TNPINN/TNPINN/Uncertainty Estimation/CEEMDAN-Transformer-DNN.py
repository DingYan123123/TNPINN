import math
import os
import time
from math import sqrt
import numpy as np
import pandas as pd
import pywt
from PyEMD import CEEMDAN
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# 参数设置
window_size = 30
forecast_steps = 10
epochs = 1000
input_size = 6
embed_dim = 32
num_heads = 4
depth = 2
learningrate = 0.01
weight_decay = 1e-5
batch_size = 32
early_stopping_patience = 50
dropout_rate = 0.2
mc_samples = 100
alpha = 0.05  # Significance level for 95% CI
rho = 2  # CWC hyperparameter
ceemdan_trials = 100  # Number of CEEMDAN trials
ceemdan_noise_std = 0.2  # Standard deviation of noise for CEEMDAN

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

# CEEMDAN去噪函数
def ceemdan_denoising(data, trials=ceemdan_trials, noise_std=ceemdan_noise_std):
    ceemdan = CEEMDAN(trials=trials, noise_std=noise_std)
    denoised_data = np.zeros_like(data)
    for i in range(data.shape[1]):  # 对每个特征进行去噪
        imfs = ceemdan(data[:, i])
        # 重构信号，排除高频噪声IMF（保留低频分量）
        denoised_data[:, i] = np.sum(imfs[1:], axis=0)  # 跳过最高频IMF
    return denoised_data

# Transformer 模型（添加 Dropout）
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, forecast_steps, depth, max_len=5000, dropout=dropout_rate):
        super(TransformerTimeSeriesModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        self.positional_encoding = nn.Parameter(self._get_positional_encoding(max_len, embed_dim), requires_grad=False)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=depth)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, forecast_steps)

    def _get_positional_encoding(self, max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mc_dropout=False):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        if mc_dropout:
            x = self.dropout(x)
        output = self.fc(x)
        return output

# 早停机制
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
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

# 归一化函数
def min_max_normalization(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-6)

# 构建时间序列样本
def build_sequences(text, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]
        target = text[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)

# 评估指标
def evaluation(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return mae, mse, rmse

# 不确定性量化指标计算
def calculate_uncertainty_metrics(y_true, ci_lower, ci_upper, alpha=0.05):
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
    model.train()
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            pred = model(input_data, mc_dropout=True)
        predictions.append(pred.detach().cpu().numpy())
    predictions = np.stack(predictions, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred
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
    fig.suptitle('Probability density of the predicted points at the 100th cycle (Transformer with CEEMDAN)',
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

# 加载并处理数据
folder_path = '../NASA Cleaned'
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

for file_name in battery_files:
    df = pd.read_csv(os.path.join(folder_path, file_name))
    features = df.iloc[:, 1:].values
    last_column = df.iloc[:, -1].values
    features = np.column_stack((features[:, 0:5], last_column / 2))
    # 应用CEEMDAN去噪
    features = ceemdan_denoising(features)
    # 去噪后归一化
    features = min_max_normalization(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)

# 模型训练和测试
maes, rmses, times = [], [], []
all_uncertainty_metrics = []
all_predictions_for_density = []
all_y_true_for_density = []
battery_names_list = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

for battery_idx, (test_battery, (test_data, test_target)) in enumerate(battery_data.items()):
    print(f"\nTesting on battery: {test_battery}")
    start_time = time.time()

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

    train_loader = DataLoader(TensorDataset(train_data_tensor, train_target_tensor),
                              batch_size=batch_size, shuffle=True)

    setup_seed(0)
    model = TransformerTimeSeriesModel(input_size, embed_dim, num_heads, forecast_steps, depth, dropout=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    train_losses = []
    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(test_data_tensor)
            val_loss = criterion(val_pred, test_target_tensor).item()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    train_end_time = time.time()
    training_time = train_end_time - train_start_time
    times.append(training_time)
    print(f"Training time for fold '{test_battery}': {training_time:.2f} seconds")

    # 用最佳模型进行预测
    model.eval()
    with torch.no_grad():
        pred = model(test_data_tensor)

    # 蒙特卡洛不确定性估计
    mean_pred, std_pred, ci_lower, ci_upper, predictions_array = mc_uncertainty_estimation(model, test_data_tensor, mc_samples)

    # 存储数据用于概率密度图
    all_predictions_for_density.append(predictions_array)
    all_y_true_for_density.append(test_target_tensor.cpu().numpy())
    battery_names_list.append(test_battery)

    # 评估性能
    pred_np = pred.cpu().numpy()
    true_np = test_target_tensor.cpu().numpy()
    mae, mse, rmse = evaluation(true_np, mean_pred)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")
    maes.append(mae * 100)
    rmses.append(rmse * 100)

    # 计算不确定性指标
    uncertainty_metrics = calculate_uncertainty_metrics(true_np, ci_lower, ci_upper, alpha)
    all_uncertainty_metrics.append(uncertainty_metrics)
    print(f"PICP: {uncertainty_metrics['PICP']:.4f}")
    print(f"PINMW: {uncertainty_metrics['PINMW']:.4f}")
    print(f"CWC: {uncertainty_metrics['CWC']:.4f}")
    print(f"95% Coverage Rate: {uncertainty_metrics['Coverage_Rate']:.2f}%")

    # 绘制单个电池的概率密度图
    if battery_idx < 4:
        plot_probability_density(predictions_array, true_np,
                                sample_idx=0, step_idx=0,
                                battery_name=f"Battery #{battery_idx + 5} (Transformer with CEEMDAN)")

    # 可视化训练损失
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"Training Loss for {test_battery} with CEEMDAN")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 可视化预测和置信区间
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 1
    plt.plot(time_steps, true_np[sample_idx], label="True", marker='o')
    plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x')
    plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                     color="red", alpha=0.2, label="95% CI")
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("SOH(%)", fontsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.show()

    # 可视化预测 vs 真实值
    plt.figure(figsize=(8, 8))
    plt.scatter(true_np.flatten(), mean_pred.flatten(), color='dodgerblue', s=60, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Prediction vs True for {test_battery} with CEEMDAN")
    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制多电池概率密度图
print("\nGenerating Figure 6 style probability density plots with CEEMDAN...")
plot_multiple_probability_densities(all_predictions_for_density[:4],
                                   all_y_true_for_density[:4],
                                   battery_names_list[:4],
                                   sample_idx=0, step_idx=0)

# 绘制不确定性指标汇总图
plot_uncertainty_metrics_summary(all_uncertainty_metrics)

# 最终汇总
print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS SUMMARY (Transformer with CEEMDAN)")
print("=" * 60)
print(f"Average RMSE: {np.mean(rmses):.3f}%")
print(f"Average MAE: {np.mean(maes):.3f}%")
print(f"Average Training Time per fold: {np.mean(times[::2]):.2f} seconds")
print(f"Average Total Time per fold: {np.mean(times[1::2]):.2f} seconds")

# 不确定性指标统计
avg_picp = np.mean([m['PICP'] for m in all_uncertainty_metrics])
avg_pinmw = np.mean([m['PINMW'] for m in all_uncertainty_metrics])
avg_cwc = np.mean([m['CWC'] for m in all_uncertainty_metrics])
avg_coverage = np.mean([m['Coverage_Rate'] for m in all_uncertainty_metrics])

print(f"\nUncertainty Quantification Metrics (Transformer with CEEMDAN):")
print(f"Average PICP: {avg_picp:.4f}")
print(f"Average PINMW: {avg_pinmw:.4f}")
print(f"Average CWC: {avg_cwc:.4f}")
print(f"Average Coverage Rate: {avg_coverage:.2f}%")

print(f"\nStandard Deviations:")
print(f"PICP Std: {np.std([m['PICP'] for m in all_uncertainty_metrics]):.4f}")
print(f"PINMW Std: {np.std([m['PINMW'] for m in all_uncertainty_metrics]):.4f}")
print(f"CWC Std: {np.std([m['CWC'] for m in all_uncertainty_metrics]):.4f}")