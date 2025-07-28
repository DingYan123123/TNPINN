import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import pywt
from torch.utils.data import DataLoader, TensorDataset, random_split

# 超参数配置
window_size = 30
forecast_steps = 10
batch_size = 32
input_size = 6
hidden_size = 32
output_size = forecast_steps
epochs = 1000
learning_rate = 0.0005
weight_decay = 0
dropout = 0.2
num_heads = 3  # 修改为 2，因为 input_size=6，需整除
mc_samples = 100  # 新增：MC 采样次数

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

# 协同 Transformer-LSTM 模型
class CollaborativeTransformerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=2, dropout=0.1):
        super(CollaborativeTransformerLSTM, self).__init__()
        self.input_size = input_size
        assert input_size % num_heads == 0, f"input_size ({input_size}) must be divisible by num_heads ({num_heads})"
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.transformer_linear = nn.Linear(input_size, 1)
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_linear = nn.Linear(hidden_size, output_size)
        self.batch_norm = nn.BatchNorm1d(input_size + 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, input, mc_dropout=False):
        transformer_output = self.transformer_encoder(input)
        transformer_soc = self.transformer_linear(transformer_output)
        combined_input = torch.cat((input, transformer_soc), dim=-1)
        batch_size, seq_len, feature_size = combined_input.shape
        combined_input = combined_input.transpose(1, 2)
        combined_input = self.batch_norm(combined_input)
        combined_input = combined_input.transpose(1, 2)
        lstm_output, _ = self.lstm(combined_input)
        final_output = self.lstm_linear(lstm_output[:, -1, :])
        return final_output

# 蒙特卡洛不确定性估计
def mc_uncertainty_estimation(model, input_data, mc_samples=100, device='cuda'):
    model.train()  # 保持 Dropout 活跃
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            pred = model(input_data, mc_dropout=True)
        predictions.append(pred.detach().cpu().numpy())
    predictions = np.stack(predictions, axis=0)  # [mc_samples, n_samples, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    return mean_pred, std_pred, ci_lower, ci_upper

# 改进的归一化方法
def improved_normalization(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (data - min_vals) / range_vals

def adaptive_wavelet_denoising(data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

def build_sequences(data, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(data) - window_size - forecast_steps + 1):
        sequence = data[i:i + window_size, :]
        target = data[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, mse, rmse

# 分步评估函数
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

class ImprovedEarlyStopping:
    def __init__(self, patience=100, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {key: val.cpu().clone() for key, val in model.state_dict().items()}
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)

def proper_train_val_split(battery_data, test_battery, val_ratio=0.2):
    train_data, train_target = [], []
    val_data, val_target = [], []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            split_idx = int(len(data) * (1 - val_ratio))
            if split_idx > 0:
                train_data.append(data[:split_idx])
                train_target.append(target[:split_idx])
            if split_idx < len(data):
                val_data.append(data[split_idx:])
                val_target.append(target[split_idx:])
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("训练集或验证集为空，请检查数据分割参数")
    return (np.concatenate(train_data), np.concatenate(train_target),
            np.concatenate(val_data), np.concatenate(val_target))

def monitor_gradients(model, epoch):
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Gradient norm: {total_norm:.4f}')
    return total_norm

def main():
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    folder_path = '../NASA Cleaned'
    battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    battery_data = {}

    for file_name in battery_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        features = df.iloc[:, 1:].values
        if features.shape[1] != input_size:
            print(f"Warning: Feature dimension mismatch in {file_name}: expected {input_size}, got {features.shape[1]}")
            continue
        features = improved_normalization(features)
        try:
            features[:, -1] = adaptive_wavelet_denoising(features[:, -1], wavelet='db4', level=3)
        except Exception as e:
            print(f"Wavelet denoising failed for {file_name}: {e}")
            continue
        data, target = build_sequences(features, window_size, forecast_steps)
        if len(data) > 0:
            battery_data[file_name] = (data, target)

    if len(battery_data) == 0:
        print("No valid battery data found!")
        return

    all_results = []
    maes, rmses = [], []

    for test_battery, (test_data, test_target) in battery_data.items():
        print(f"\nTesting on battery: {test_battery}")
        try:
            train_data, train_target, val_data, val_target = proper_train_val_split(
                battery_data, test_battery, val_ratio=0.2
            )
            train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
            train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
            val_data_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)
            val_target_tensor = torch.tensor(val_target, dtype=torch.float32).to(device)
            test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
            test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

            train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
            val_dataset = TensorDataset(val_data_tensor, val_target_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            setup_seed(42)
            model = CollaborativeTransformerLSTM(
                input_size, hidden_size, output_size,
                num_layers=2, num_heads=num_heads, dropout=dropout
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15, verbose=False
            )
            early_stopping = ImprovedEarlyStopping(patience=50, min_delta=1e-6)

            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                model.train()
                epoch_train_loss = 0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs, mc_dropout=True)  # 训练时启用 Dropout
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_train_loss += loss.item()
                avg_train_loss = epoch_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model(inputs, mc_dropout=False)
                        loss = criterion(outputs, targets)
                        epoch_val_loss += loss.item()
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                scheduler.step(avg_val_loss)
                monitor_gradients(model, epoch)
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # 测试阶段
            model.eval()
            with torch.no_grad():
                test_pred = model(test_data_tensor, mc_dropout=False)
            test_pred_np = test_pred.cpu().numpy()
            test_true_np = test_target_tensor.cpu().numpy()

            # MC 不确定性估计
            mean_pred, std_pred, ci_lower, ci_upper = mc_uncertainty_estimation(model, test_data_tensor, mc_samples, device)

            # 计算覆盖概率
            coverage = np.mean((test_true_np >= ci_lower) & (test_true_np <= ci_upper))
            print(f"95% Coverage Probability: {coverage * 100:.2f}%")

            # 分步评估
            results = evaluation_improved(test_true_np, test_pred_np)
            all_results.append(results)
            print(f"\n{test_battery} Overall Results:")
            print(f"Overall MAE: {results['overall_mae'] * 100:.3f}%")
            print(f"Overall RMSE: {results['overall_rmse'] * 100:.3f}%")

            maes.append(results['overall_mae'])
            rmses.append(results['overall_rmse'])

            # 绘制训练和验证损失曲线
            plt.figure(figsize=(12, 6))
            plt.plot(train_losses, label='训练损失', alpha=0.7)
            plt.plot(val_losses, label='验证损失', alpha=0.7)
            plt.xlabel('轮次', fontsize=18)
            plt.ylabel('MSE 损失', fontsize=18)
            plt.title(f'训练和验证曲线 - {test_battery}', fontsize=18)
            plt.legend(fontsize=18)
            plt.grid(True)
            plt.tick_params(axis='both', labelsize=20)
            plt.yscale('log')
            plt.show()

            # 绘制预测和置信区间（示例：第一个样本）
            plt.figure(figsize=(12, 6))
            time_steps = np.arange(forecast_steps)
            sample_idx = 30
            plt.plot(time_steps, test_true_np[sample_idx], label="True", marker='o')
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
            test_true_flat = test_true_np.flatten()
            test_pred_flat = test_pred_np.flatten()
            plt.scatter(test_true_flat, test_pred_flat, s=100, color='dodgerblue', alpha=0.8)
            plt.plot([min(test_true_flat), max(test_true_flat)], [min(test_true_flat), max(test_true_flat)],
                     'r--', label='理想线')
            plt.xlabel("真实值", fontsize=18)
            plt.ylabel("预测值", fontsize=18)
            plt.title(f"预测 vs 真实值 - {test_battery}", fontsize=18)
            plt.legend(fontsize=18)
            plt.grid(True)
            plt.tick_params(axis='both', labelsize=20)
            plt.show()

        except Exception as e:
            print(f"Error processing {test_battery}: {e}")
            continue

    # 输出总体结果
    if len(maes) > 0 and len(rmses) > 0:
        print(f"\n=== Cross-validation results ===")
        print(f"Average RMSE: {np.mean(rmses) * 100:.3f}%")
        print(f"Average MAE: {np.mean(maes) * 100:.3f}%")
        print(f"RMSE std: {np.std(rmses) * 100:.3f}%")
        print(f"MAE std: {np.std(maes) * 100:.3f}%")
    else:
        print("No valid results obtained!")

if __name__ == "__main__":
    main()