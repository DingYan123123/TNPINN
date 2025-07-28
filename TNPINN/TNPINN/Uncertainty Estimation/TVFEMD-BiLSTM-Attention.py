import os
import numpy as np
import pandas as pd
from math import sqrt
import pywt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert, butter, filtfilt
from PyEMD import EMD, CEEMDAN
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ------------------ 配置 ------------------
window_size = 30
forecast_steps = 10
batch_size = 32
input_size = 6
hidden_size = 64
output_size = forecast_steps
epochs = 1000
learningrate = 0.002
weight_decay = 0
folder_path = '../NASA Cleaned'
patience = 100
min_delta = 0
dropout_rate = 0.2  # 已有的 Dropout 率
mc_samples = 100  # 新增：MC 采样次数

# ------------------ 设置随机种子 ------------------
def setup_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ------------------ 优化后的 TVFEMD 类 ------------------
class TVFEMD:
    def __init__(self, fs=1.0, n_imfs=5, noise_std=0.01, n_trials=30, max_imf=10, adaptive_filter=True):
        self.fs = fs
        self.n_imfs = n_imfs
        self.noise_std = noise_std
        self.n_trials = n_trials
        self.max_imf = max_imf
        self.adaptive_filter = adaptive_filter

    def instantaneous_frequency(self, signal):
        analytic_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.fs
        instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])
        return instantaneous_frequency

    def adaptive_bandpass_filter(self, signal, center_freq, bandwidth_ratio=0.4):
        if len(signal) < 10:
            return signal
        inst_freq = self.instantaneous_frequency(signal)
        window_size = min(len(inst_freq) // 10, 50)
        if window_size > 1:
            inst_freq_smooth = np.convolve(inst_freq, np.ones(window_size) / window_size, mode='same')
        else:
            inst_freq_smooth = inst_freq
        filtered_signal = np.zeros_like(signal)
        for i in range(len(signal)):
            if i < len(signal) // 4:
                local_freq = center_freq
            else:
                local_freq = max(0.01, min(0.49, inst_freq_smooth[i]))
            bandwidth = local_freq * bandwidth_ratio
            lowcut = max(0.01, local_freq - bandwidth / 2)
            highcut = min(0.49, local_freq + bandwidth / 2)
            if lowcut >= highcut:
                lowcut = max(0.01, local_freq * 0.7)
                highcut = min(0.49, local_freq * 1.3)
            window_start = max(0, i - window_size // 2)
            window_end = min(len(signal), i + window_size // 2)
            local_signal = signal[window_start:window_end]
            if len(local_signal) > 6:
                try:
                    nyq = 0.5 * self.fs
                    low, high = lowcut / nyq, highcut / nyq
                    b, a = butter(3, [low, high], btype='band')
                    filtered_local = filtfilt(b, a, local_signal)
                    center_idx = len(filtered_local) // 2
                    filtered_signal[i] = filtered_local[center_idx]
                except:
                    filtered_signal[i] = signal[i]
            else:
                filtered_signal[i] = signal[i]
        return filtered_signal

    def compute_frequency_content(self, imf):
        if len(imf) < 10:
            return 0.1
        fft_vals = np.fft.fft(imf)
        freqs = np.fft.fftfreq(len(imf), 1 / self.fs)
        positive_freqs = freqs[:len(freqs) // 2]
        positive_fft = np.abs(fft_vals[:len(freqs) // 2])
        if len(positive_fft) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            dominant_freq = positive_freqs[dominant_freq_idx]
            return abs(dominant_freq)
        return 0.1

    def imf_selection_criterion(self, imfs):
        if len(imfs) == 0:
            return []
        scores = []
        for i, imf in enumerate(imfs):
            energy_density = np.sum(imf ** 2) / len(imf)
            freq_content = self.compute_frequency_content(imf)
            variance_contribution = np.var(imf) / np.sum([np.var(imf_j) for imf_j in imfs])
            signal_power = np.mean(imf ** 2)
            noise_power = np.var(np.diff(imf))
            snr = signal_power / (noise_power + 1e-10)
            score = (0.3 * energy_density + 0.2 * freq_content +
                     0.3 * variance_contribution + 0.2 * np.log10(snr + 1))
            scores.append(score)
        return np.argsort(scores)[::-1]

    def decompose(self, signal):
        if len(signal) < 10:
            return signal
        ceemdan = CEEMDAN(trials=self.n_trials, noise_std=self.noise_std)
        try:
            imfs = ceemdan(signal, max_imf=self.max_imf)
        except:
            emd = EMD()
            imfs = emd(signal, max_imf=self.max_imf)
        if len(imfs) == 0:
            return signal
        selected_indices = self.imf_selection_criterion(imfs)[:self.n_imfs]
        filtered_imfs = []
        for idx in selected_indices:
            if idx < len(imfs):
                imf = imfs[idx]
                if self.adaptive_filter:
                    dominant_freq = self.compute_frequency_content(imf)
                    filtered_imf = self.adaptive_bandpass_filter(imf, dominant_freq)
                else:
                    filtered_imf = imf
                filtered_imfs.append(filtered_imf)
        if len(filtered_imfs) > 0:
            reconstructed = np.sum(filtered_imfs, axis=0)
        else:
            reconstructed = signal
        return reconstructed

# ------------------ 应用小波去噪和 TVFEMD ------------------
def apply_tvfemd_to_battery(raw_features, fs=1.0, **kwargs):
    tvfemd = TVFEMD(fs=fs, n_imfs=5, noise_std=0.01, n_trials=30, max_imf=10, adaptive_filter=True)
    num_samples, num_features = raw_features.shape
    processed_features = np.zeros_like(raw_features)
    for i in range(num_features):
        feature_data = raw_features[:, i]
        if len(feature_data) > 10:
            try:
                denoised_data = wavelet_denoising(feature_data, wavelet='db4', level=4, threshold=0.1)
                coeffs = np.polyfit(np.arange(len(denoised_data)), denoised_data, 1)
                trend = np.polyval(coeffs, np.arange(len(denoised_data)))
                detrended = denoised_data - trend
                processed = tvfemd.decompose(detrended)
                processed_features[:, i] = processed + trend
            except Exception as e:
                print(f"处理特征 {i} 时出错: {str(e)}，使用原始数据")
                processed_features[:, i] = feature_data
        else:
            print(f"特征 {i} 数据长度不足，跳过去噪和 TVFEMD 处理")
            processed_features[:, i] = feature_data
    return processed_features

# ------------------ 小波去噪函数 ------------------
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

# ------------------ BiLSTM 模型 ------------------
class ImprovedAttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=2, dropout=dropout_rate):
        super(ImprovedAttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.compress = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4,
                                               dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, out_size)
        self.relu = nn.ReLU()

    def forward(self, x, mc_dropout=False):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.compress(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = attn_out[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        if mc_dropout:
            output = self.dropout(output)
        output = self.fc2(output)
        return output

# ------------------ 蒙特卡洛不确定性估计 ------------------
def mc_uncertainty_estimation(model, input_data, mc_samples=100, device='cuda'):
    model.train()  # 保持 Dropout 活跃
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            pred = model(input_data, mc_dropout=True)
        predictions.append(pred.detach().cpu().numpy())
    predictions = np.stack(predictions, axis=0)  # 形状为 [mc_samples, n_samples, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)  # 形状为 [n_samples, forecast_steps]
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    return mean_pred, std_pred, ci_lower, ci_upper

# ------------------ 数据预处理器 ------------------
class DataPreprocessor:
    def __init__(self):
        self.scalers = {}

    def fit_transform(self, data, battery_name):
        scaler = StandardScaler()
        transformed = scaler.fit_transform(data)
        self.scalers[battery_name] = scaler
        return transformed

    def transform(self, data, battery_name):
        if battery_name in self.scalers:
            return self.scalers[battery_name].transform(data)
        else:
            raise ValueError(f"No scaler found for battery: {battery_name}")

# ------------------ 序列构造 ------------------
def build_sequences_simple(data, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(data) - window_size - forecast_steps):
        x.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + forecast_steps, -1])
    x, y = np.array(x), np.array(y)
    return x, y

# ------------------ 改进的评估函数 ------------------
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

# ------------------ 打印分步结果汇总 ------------------
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

# ------------------ 交叉验证分步结果统计 ------------------
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

# ------------------ 早停类 ------------------
class EarlyStopping:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_state(self, model):
        model.load_state_dict(self.best_state)

# ------------------ 主流程 ------------------
def main():
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在，请检查路径")
        return 0

    battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not battery_files:
        print("未找到 CSV 文件")
        return 0

    battery_data = {}
    preprocessor = DataPreprocessor()

    print("正在加载和预处理数据...")
    for file in battery_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, file))
            raw_features = df.iloc[:, 1:].values
            print(f"对 {file} 应用小波去噪和 TVFEMD 到所有特征...")
            raw_features = apply_tvfemd_to_battery(raw_features, fs=1.0)
            norm_features = preprocessor.fit_transform(raw_features, file)
            train_x, train_y = build_sequences_simple(norm_features, window_size, forecast_steps)
            battery_data[file] = {'train_x': train_x, 'train_y': train_y}
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

    all_results = []

    for test_battery in battery_data.keys():
        print(f"\n{'=' * 60}")
        print(f"测试电池: {test_battery}")
        print('=' * 60)

        train_x_list, train_y_list = [], []
        for battery_name, data in battery_data.items():
            if battery_name != test_battery:
                train_x_list.append(data['train_x'])
                train_y_list.append(data['train_y'])
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        test_data = battery_data[test_battery]
        test_x = test_data['train_x']
        test_y = test_data['train_y']
        val_x, val_y = test_x, test_y

        print(f"训练集大小: {train_x.shape}, 验证/测试集大小: {val_x.shape}")

        train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                                     torch.tensor(train_y, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(torch.tensor(val_x, dtype=torch.float32),
                                    torch.tensor(val_y, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = ImprovedAttentionBiLSTM(input_size, hidden_size, output_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learningrate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        train_losses = []
        val_losses = []

        print("开始训练...")
        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x, mc_dropout=True)  # 训练时启用 Dropout
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_loss = np.mean(batch_losses)
            train_losses.append(epoch_loss)

            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x, mc_dropout=False)
                    loss = criterion(outputs, batch_y)
                    val_batch_losses.append(loss.item())
            val_loss = np.mean(val_batch_losses)
            val_losses.append(val_loss)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"早停触发，停止于轮次 {epoch}")
                early_stopping.load_best_state(model)
                break
            scheduler.step()

            if epoch % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}")

        # 测试阶段
        model.eval()
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
        with torch.no_grad():
            test_pred = model(test_x_tensor, mc_dropout=False).cpu().numpy()
        test_true = test_y

        # MC 不确定性估计
        mean_pred, std_pred, ci_lower, ci_upper = mc_uncertainty_estimation(model, test_x_tensor, mc_samples, device)

        # 计算覆盖概率
        coverage = np.mean((test_true >= ci_lower) & (test_true <= ci_upper))
        print(f"95% 覆盖概率: {coverage * 100:.2f}%")

        # 评估结果
        results = evaluation_improved(test_true, test_pred)
        all_results.append(results)
        print(f"\n{test_battery} 整体测试结果:")
        print(f"整体 MAE: {results['overall_mae'] * 100:.3f}%")
        print(f"整体 RMSE: {results['overall_rmse'] * 100:.3f}%")

        # 绘制训练和验证曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失', alpha=0.7)
        plt.plot(val_losses, label='验证损失', alpha=0.7)
        plt.xlabel('轮次', fontsize=18)
        plt.ylabel('MSE 损失', fontsize=18)
        plt.title(f'训练和验证曲线 - {test_battery}', fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=20)
        plt.yscale('log')

        # 绘制预测和置信区间（示例：第一个样本）
        plt.figure(figsize=(12, 6))
        time_steps = np.arange(forecast_steps)
        sample_idx = 0
        plt.plot(time_steps, test_true[sample_idx], label="True", marker='o')
        plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x')
        plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                         color="red", alpha=0.2, label="95% CI")
        # plt.title(f"预测与 95% 置信区间 - {test_battery}", fontsize=18)
        plt.xlabel("Time Step", fontsize=18)
        plt.ylabel("SOH(%)", fontsize=18)
        plt.tick_params(axis='both', labelsize=20)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()

        # 绘制预测与真实值的散点图
        plt.figure(figsize=(8, 8))
        test_true_flat = test_true.flatten()
        test_pred_flat = test_pred.flatten()
        plt.scatter(test_true_flat, test_pred_flat, s=100, color='dodgerblue', alpha=0.8)
        plt.plot([min(test_true_flat), max(test_true_flat)], [min(test_true_flat), max(test_true_flat)],
                 'r--', label='Ideal')
        plt.xlabel("True", fontsize=18)
        plt.ylabel("Predicted", fontsize=18)
        # plt.title(f"预测 vs 真实值 - {test_battery}", fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=20)
        plt.show()

    # 交叉验证总结
    print_stepwise_summary(all_results, battery_files)
    print_cross_validation_stepwise_summary(all_results, battery_files)

    print("\n=== 交叉验证总结 ===")
    overall_maes = [result['overall_mae'] for result in all_results]
    overall_rmses = [result['overall_rmse'] for result in all_results]
    print(f"平均 MAE: {np.mean(overall_maes) * 100:.3f}% ± {np.std(overall_maes) * 100:.3f}%")
    print(f"平均 RMSE: {np.mean(overall_rmses) * 100:.3f}% ± {np.std(overall_rmses) * 100:.3f}%")

    return 0

if __name__ == "__main__":
    results = main()