import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pywt
import seaborn as sns
from scipy.stats import gaussian_kde

# 超参数配置
window_size = 10
forecast_steps = 10
input_size = 60  # window_size * num_features (10 * 6)
hidden_size = 64
HPM_hidden_size = 4
output_size = forecast_steps
epochs = 1000
weight_decay = 0
learningrate = 0.001
batch_size = 32
mc_samples = 100  # MC 采样次数
alpha = 0.05  # Significance level for 95% CI
rho = 2  # CWC hyperparameter
dropout_rate = 0.2  # Dropout rate for uncertainty estimation

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

# 线性模型（带 PINN 和 Dropout）
class Linear(nn.Module):
    def __init__(self, input_size, hidden_size, HPM_hidden_size, output_size, dropout_rate=0.2):
        super(Linear, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.hpm = DeepHPM(output_size + input_size, HPM_hidden_size, input_size, dropout_rate)

    def forward(self, input, mc_dropout=False):
        input = input.requires_grad_(True)
        hidden = self.linear1(input)
        if mc_dropout:
            hidden = self.dropout(hidden)
        hidden = self.tanh(hidden)
        output = self.linear2(hidden)
        if mc_dropout:
            output = self.dropout(output)
        output_input = torch.autograd.grad(
            output, input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        HPM_input = torch.cat((input, output), dim=2)
        G = self.hpm(HPM_input, mc_dropout)
        F = output_input - G
        HPM_input = HPM_input.requires_grad_(True)
        F_out = torch.autograd.grad(
            F, input,
            grad_outputs=torch.ones_like(F),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        return output, F, F_out

# DeepHPM 模型（带 Dropout）
class DeepHPM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(DeepHPM, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, mc_dropout=False):
        x = self.layer1(x)
        x = self.tanh(x)
        if mc_dropout:
            x = self.dropout(x)
        x = self.layer2(x)
        return x

# 蒙特卡洛不确定性估计
def mc_uncertainty_estimation(model, input_data, mc_samples=100, device='cuda'):
    model.train()
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            output, _, _ = model(input_data, mc_dropout=True)
        predictions.append(output.detach().cpu().numpy())
    predictions = np.stack(predictions, axis=0)  # [mc_samples, n_samples, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    return mean_pred, std_pred, ci_lower, ci_upper, predictions

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

# 改进的评估函数
def evaluation_improved(y_true, y_pred):
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = sqrt(mse)
    step_wise_mae = []
    step_wise_rmse = []
    print("\n=== 分步预测结果 ===")
    print(f"{'步长':<6} {'MAE':<12} {'RMSE':<12}")
    print("-" * 32)
    for step in range(y_true.shape[1]):
        step_mae = mean_absolute_error(y_true[:, step], y_pred[:, step])
        step_mse = mean_squared_error(y_true[:, step], y_pred[:, step])
        step_rmse = sqrt(step_mse)
        step_wise_mae.append(step_mae)
        step_wise_rmse.append(step_rmse)
        print(f"第{step + 1:2d}步  {step_mae * 100:8.3f}%  {step_rmse * 100:8.3f}%")
    print("-" * 32)
    print(f"{'平均':<6} {np.mean(step_wise_mae) * 100:8.3f}%  {np.mean(step_wise_rmse) * 100:8.3f}%")
    return {
        'overall_mae': mae,
        'overall_rmse': rmse,
        'step_wise_mae': step_wise_mae,
        'step_wise_rmse': step_wise_rmse
    }

# 打印分步结果汇总
def print_stepwise_summary(all_results, battery_files):
    print("\n" + "=" * 60)
    print("所有电池分步预测结果汇总")
    print("=" * 60)
    header = f"{'电池':<20} {'步长':<6} {'MAE(%)':<10} {'RMSE(%)':<10}"
    print(header)
    print("-" * 60)
    for i, (battery_file, results) in enumerate(zip(battery_files, all_results)):
        battery_name = battery_file.replace('.csv', '')
        for step in range(len(results['step_wise_mae'])):
            mae_pct = results['step_wise_mae'][step] * 100
            rmse_pct = results['step_wise_rmse'][step] * 100
            if step == 0:
                print(f"{battery_name:<20} 第{step + 1:2d}步  {mae_pct:8.3f}  {rmse_pct:8.3f}")
            else:
                print(f"{'':<20} 第{step + 1:2d}步  {mae_pct:8.3f}  {rmse_pct:8.3f}")
        avg_mae = np.mean(results['step_wise_mae']) * 100
        avg_rmse = np.mean(results['step_wise_rmse']) * 100
        print(f"{'':<20} {'平均':<6} {avg_mae:8.3f}  {avg_rmse:8.3f}")
        print("-" * 60)

# 交叉验证分步结果统计
def print_cross_validation_stepwise_summary(all_results, battery_files):
    print("\n" + "=" * 80)
    print("交叉验证分步预测统计结果")
    print("=" * 80)
    step_wise_maes = np.array([result['step_wise_mae'] for result in all_results])
    step_wise_rmses = np.array([result['step_wise_rmse'] for result in all_results])
    mean_mae_per_step = np.mean(step_wise_maes, axis=0)
    std_mae_per_step = np.std(step_wise_maes, axis=0)
    min_mae_per_step = np.min(step_wise_maes, axis=0)
    max_mae_per_step = np.max(step_wise_maes, axis=0)
    mean_rmse_per_step = np.mean(step_wise_rmses, axis=0)
    std_rmse_per_step = np.std(step_wise_rmses, axis=0)
    min_rmse_per_step = np.min(step_wise_rmses, axis=0)
    max_rmse_per_step = np.max(step_wise_rmses, axis=0)
    print(f"{'步长':<6} {'MAE统计':<40} {'RMSE统计':<40}")
    print(f"{'':6} {'均值±标准差':<20} {'最小值':<10} {'最大值':<10} {'均值±标准差':<20} {'最小值':<10} {'最大值':<10}")
    print("-" * 80)
    for step in range(len(mean_mae_per_step)):
        mae_mean_std = f"{mean_mae_per_step[step] * 100:.3f}±{std_mae_per_step[step] * 100:.3f}"
        mae_min = f"{min_mae_per_step[step] * 100:.3f}"
        mae_max = f"{max_mae_per_step[step] * 100:.3f}"
        rmse_mean_std = f"{mean_rmse_per_step[step] * 100:.3f}±{std_rmse_per_step[step] * 100:.3f}"
        rmse_min = f"{min_rmse_per_step[step] * 100:.3f}"
        rmse_max = f"{max_rmse_per_step[step] * 100:.3f}"
        print(f"第{step + 1:2d}步  {mae_mean_std:<20} {mae_min:<10} {mae_max:<10} {rmse_mean_std:<20} {rmse_min:<10} {rmse_max:<10}")
    print("-" * 80)
    overall_mean_mae = np.mean(mean_mae_per_step) * 100
    overall_mean_rmse = np.mean(mean_rmse_per_step) * 100
    overall_std_mae = np.std(mean_mae_per_step) * 100
    overall_std_rmse = np.std(mean_rmse_per_step) * 100
    print(f"{'总体':<6} {overall_mean_mae:.3f}±{overall_std_mae:.3f} %{'':<8} {'':<10} {overall_mean_rmse:.3f}±{overall_std_rmse:.3f} %{'':<8} {'':<10}")
    return {
        'mean_mae_per_step': mean_mae_per_step,
        'std_mae_per_step': std_mae_per_step,
        'min_mae_per_step': min_mae_per_step,
        'max_mae_per_step': max_mae_per_step,
        'mean_rmse_per_step': mean_rmse_per_step,
        'std_rmse_per_step': std_rmse_per_step,
        'min_rmse_per_step': min_rmse_per_step,
        'max_rmse_per_step': max_rmse_per_step
    }

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
    fig.suptitle('Probability density of the predicted points at the 100th cycle (PINN+XGBoost)',
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

# 归一化函数
def min_max_normalization_per_feature(data):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    min_vals = torch.min(data, dim=0)[0]
    max_vals = torch.max(data, dim=0)[0]
    return (data - min_vals) / (max_vals - min_vals + 1e-6)

def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

def build_sequences(text, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]
        target = text[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)

class EarlyStopping:
    def __init__(self, patience=100, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_weights = {key: val.cpu().clone() for key, val in model.state_dict().items()}
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            model.load_state_dict(self.best_weights)
            return True
        return False

def main():
    setup_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    folder_path = '../NASA Cleaned'
    battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    battery_data = {}
    all_uncertainty_metrics = []
    all_predictions_for_density = []
    all_y_true_for_density = []
    battery_names_list = []
    all_results = []
    maes, rmses = [], []

    # 数据加载和处理
    for file_name in battery_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        features = df.iloc[:, 1:].values
        last_column = df.iloc[:, -1].values
        last_column = np.apply_along_axis(wavelet_denoising, 0, last_column)
        features = np.column_stack((features[:, 0:5], last_column / 2))
        features = min_max_normalization_per_feature(features)
        data, target = build_sequences(features, window_size, forecast_steps)
        if len(data) > 0:
            battery_data[file_name] = (data, target)

    if len(battery_data) == 0:
        print("No valid battery data found!")
        return

    for battery_idx, (test_battery, (test_data, test_target)) in enumerate(battery_data.items()):
        print(f"\nTesting on battery: {test_battery}")
        try:
            train_data = []
            train_target = []
            for battery, (data, target) in battery_data.items():
                if battery != test_battery:
                    train_data.append(data)
                    train_target.append(target)
            train_data = np.concatenate(train_data)
            train_target = np.concatenate(train_target)

            train_data_tensor = torch.tensor(train_data, requires_grad=True, dtype=torch.float32).to(device)
            train_data_tensor = train_data_tensor.reshape(len(train_data_tensor), 1, -1)
            train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
            test_data_tensor = torch.tensor(test_data, requires_grad=True, dtype=torch.float32).to(device)
            test_data_tensor = test_data_tensor.reshape(len(test_data_tensor), 1, -1)
            test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

            model = Linear(input_size, hidden_size, HPM_hidden_size, output_size, dropout_rate).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)
            early_stopping = EarlyStopping(patience=100, delta=0.01)

            train_losses = []
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output, Fx, Fxs = model(train_data_tensor, mc_dropout=True)
                loss = criterion(output, train_target_tensor) + torch.mean(Fx ** 2) + torch.mean(Fxs ** 2)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                model.eval()
                with torch.no_grad():
                    pred, _, _ = model(test_data_tensor, mc_dropout=False)
                pred_np = pred.detach().squeeze().cpu().numpy()
                test_np = test_target_tensor.detach().squeeze().cpu().numpy()
                val_loss = np.mean((pred_np - test_np) ** 2)
                if early_stopping(val_loss, model):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

            # MC 不确定性估计
            mean_pred, std_pred, ci_lower, ci_upper, predictions_array = mc_uncertainty_estimation(model, test_data_tensor, mc_samples, device)

            # 存储数据用于概率密度图
            all_predictions_for_density.append(predictions_array)
            all_y_true_for_density.append(test_np)
            battery_names_list.append(test_battery)

            # 计算不确定性指标
            uncertainty_metrics = calculate_uncertainty_metrics(test_np, ci_lower, ci_upper, alpha)
            all_uncertainty_metrics.append(uncertainty_metrics)
            print(f"PICP: {uncertainty_metrics['PICP']:.4f}")
            print(f"PINMW: {uncertainty_metrics['PINMW']:.4f}")
            print(f"CWC: {uncertainty_metrics['CWC']:.4f}")
            print(f"95% Coverage Rate: {uncertainty_metrics['Coverage_Rate']:.2f}%")

            # 分步评估
            results = evaluation_improved(test_np, mean_pred)
            all_results.append(results)
            print(f"\n{test_battery} Overall Results:")
            print(f"Overall MAE: {results['overall_mae'] * 100:.3f}%")
            print(f"Overall RMSE: {results['overall_rmse'] * 100:.3f}%")

            maes.append(results['overall_mae'] * 100)
            rmses.append(results['overall_rmse'] * 100)

            # 绘制单个电池的概率密度图
            if battery_idx < 4:
                plot_probability_density(predictions_array, test_np,
                                        sample_idx=0, step_idx=0,
                                        battery_name=f"Battery #{battery_idx + 5} (PINN+XGBoost)")

            # 绘制训练损失图
            plt.figure(figsize=(12, 6))
            plt.plot(train_losses, label="Training Loss", alpha=0.7)
            plt.xlabel('Epochs', fontsize=18)
            plt.ylabel('Loss', fontsize=18)
            plt.title(f"Training Loss over Epochs for {test_battery}", fontsize=18)
            plt.legend(fontsize=18)
            plt.grid(True)
            plt.tick_params(axis='both', labelsize=20)
            plt.yscale('log')
            plt.show()

            # 绘制预测和置信区间
            plt.figure(figsize=(12, 6))
            time_steps = np.arange(forecast_steps)
            sample_idx = 0
            plt.plot(time_steps, test_np[sample_idx], label="True", marker='o')
            plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x')
            plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                             color="red", alpha=0.2, label="95% CI")
            plt.xlabel("Time Step", fontsize=18)
            plt.ylabel("SOH(%)", fontsize=18)
            plt.tick_params(axis='both', labelsize=20)
            plt.legend(fontsize=18)
            plt.grid(True)
            plt.show()

            # 绘制预测与真实值的散点图
            plt.figure(figsize=(8, 8))
            test_np_flat = test_np.flatten()
            pred_np_flat = mean_pred.flatten()
            plt.scatter(test_np_flat, pred_np_flat, s=100, color='dodgerblue', alpha=0.8)
            plt.plot([min(test_np_flat), max(test_np_flat)], [min(test_np_flat), max(test_np_flat)],
                     'r--', label='Ideal')
            plt.xlabel("True Values", fontsize=18)
            plt.ylabel("Predicted Values", fontsize=18)
            plt.title(f"Predicted vs True Values - {test_battery}", fontsize=18)
            plt.legend(fontsize=18)
            plt.grid(True)
            plt.tick_params(axis='both', labelsize=20)
            plt.show()

        except Exception as e:
            print(f"Error processing {test_battery}: {e}")
            continue

    # 绘制多电池概率密度图
    print("\nGenerating Figure 6 style probability density plots...")
    plot_multiple_probability_densities(all_predictions_for_density[:4],
                                       all_y_true_for_density[:4],
                                       battery_names_list[:4],
                                       sample_idx=0, step_idx=0)

    # 绘制不确定性指标汇总图
    plot_uncertainty_metrics_summary(all_uncertainty_metrics)

    # 输出分步结果汇总
    print_stepwise_summary(all_results, battery_files)
    cv_stats = print_cross_validation_stepwise_summary(all_results, battery_files)

    # 输出总体结果
    if len(maes) > 0 and len(rmses) > 0:
        print(f"\n=== Cross-validation results ===")
        print(f"Average RMSE: {np.mean(rmses):.3f}%")
        print(f"Average MAE: {np.mean(maes):.3f}%")
        print(f"RMSE std: {np.std(rmses):.3f}%")
        print(f"MAE std: {np.std(maes):.3f}%")

        # 不确定性指标统计
        avg_picp = np.mean([m['PICP'] for m in all_uncertainty_metrics])
        avg_pinmw = np.mean([m['PINMW'] for m in all_uncertainty_metrics])
        avg_cwc = np.mean([m['CWC'] for m in all_uncertainty_metrics])
        avg_coverage = np.mean([m['Coverage_Rate'] for m in all_uncertainty_metrics])

        print(f"\nUncertainty Quantification Metrics (PINN+XGBoost):")
        print(f"Average PICP: {avg_picp:.4f}")
        print(f"Average PINMW: {avg_pinmw:.4f}")
        print(f"Average CWC: {avg_cwc:.4f}")
        print(f"Average Coverage Rate: {avg_coverage:.2f}%")

        print(f"\nStandard Deviations:")
        print(f"PICP Std: {np.std([m['PICP'] for m in all_uncertainty_metrics]):.4f}")
        print(f"PINMW Std: {np.std([m['PINMW'] for m in all_uncertainty_metrics]):.4f}")
        print(f"CWC Std: {np.std([m['CWC'] for m in all_uncertainty_metrics]):.4f}")
    else:
        print("No valid results obtained!")

if __name__ == "__main__":
    main()