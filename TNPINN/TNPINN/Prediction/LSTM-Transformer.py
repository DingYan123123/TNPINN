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
window_size = 10  # 设置窗口大小
forecast_steps = 10  # 预测步数
batch_size = 32  # 增加批次大小
input_size = 6  # 输入特征数（需与数据特征数一致）
hidden_size = 32  # 隐藏层大小
output_size = forecast_steps  # 输出层大小
epochs = 1000  # 训练轮数
learning_rate = 0.0005  # 降低学习率
weight_decay = 0 # 减少L2正则化强度
dropout = 0.1  # 增加dropout
num_heads = 3  # 确保能被input_size整除


# 设置随机种子以确保可重复性
def setup_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# 修复后的协同Transformer-LSTM模型
class CollaborativeTransformerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=6, dropout=0.1):
        super(CollaborativeTransformerLSTM, self).__init__()
        self.input_size = input_size

        # 确保input_size能被num_heads整除
        assert input_size % num_heads == 0, f"input_size ({input_size}) must be divisible by num_heads ({num_heads})"

        # Transformer部分 - 直接处理原始输入
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.transformer_linear = nn.Linear(input_size, 1)  # 输出SOC估计值

        # LSTM部分
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_linear = nn.Linear(hidden_size, output_size)

        # 添加BatchNorm层
        self.batch_norm = nn.BatchNorm1d(input_size + 1)

        # 权重初始化
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

    def forward(self, input):
        # Transformer处理原始输入
        transformer_output = self.transformer_encoder(input)  # [batch, seq_len, input_size]
        transformer_soc = self.transformer_linear(transformer_output)  # [batch, seq_len, 1]

        # 拼接原始输入和Transformer的SOC估计
        combined_input = torch.cat((input, transformer_soc), dim=-1)  # [batch, seq_len, input_size + 1]

        # BatchNorm (需要转换维度)
        batch_size, seq_len, feature_size = combined_input.shape
        combined_input = combined_input.transpose(1, 2)  # [batch, feature_size, seq_len]
        combined_input = self.batch_norm(combined_input)
        combined_input = combined_input.transpose(1, 2)  # [batch, seq_len, feature_size]

        # LSTM处理
        lstm_output, _ = self.lstm(combined_input)  # [batch, seq_len, hidden_size]
        final_output = self.lstm_linear(lstm_output[:, -1, :])  # 取最后一个时间步的输出 [batch, output_size]

        return final_output


# 改进的归一化方法
def improved_normalization(data):
    """改进的归一化方法，避免过度归一化"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    # 避免除零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (data - min_vals) / range_vals


def z_score_standardization(data):
    """Z-score标准化"""
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    std_vals[std_vals == 0] = 1  # 避免除零
    return (data - mean_vals) / std_vals


def adaptive_wavelet_denoising(data, wavelet='db4', level=3):
    """自适应小波去噪"""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # 使用更保守的阈值策略
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


def build_sequences(data, window_size, forecast_steps):
    """构建序列数据"""
    x, y = [], []
    for i in range(len(data) - window_size - forecast_steps + 1):
        sequence = data[i:i + window_size, :]
        target = data[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)


def evaluation(y_test, y_predict):
    """评估指标"""
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, mse, rmse


class ImprovedEarlyStopping:
    """改进的早停策略"""

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
    """正确的训练验证分割"""
    train_data, train_target = [], []
    val_data, val_target = [], []

    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            # 按时间顺序分割，避免数据泄露
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
    """监控梯度情况"""
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


# 主程序
def main():
    # 数据加载和处理
    folder_path = '../NASA Cleaned'  # 替换为电池文件所在的文件夹路径
    battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    battery_data = {}

    # 对每个电池的数据进行处理
    for file_name in battery_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # 提取特征
        features = df.iloc[:, 1:].values
        if features.shape[1] != input_size:
            print(f"Warning: Feature dimension mismatch in {file_name}: expected {input_size}, got {features.shape[1]}")
            continue

        # 改进的数据预处理
        features = improved_normalization(features)

        # 对容量列进行小波去噪
        try:
            features[:, -1] = adaptive_wavelet_denoising(features[:, -1], wavelet='db4', level=3)
        except Exception as e:
            print(f"Wavelet denoising failed for {file_name}: {e}")
            continue

        # 构建序列
        data, target = build_sequences(features, window_size, forecast_steps)
        if len(data) > 0:
            battery_data[file_name] = (data, target)

    if len(battery_data) == 0:
        print("No valid battery data found!")
        return

    # 交叉验证
    maes, rmses = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 训练过程
    for test_battery, (test_data, test_target) in battery_data.items():
        print(f"\nTesting on battery: {test_battery}")

        try:
            # 正确的数据分割
            train_data, train_target, val_data, val_target = proper_train_val_split(
                battery_data, test_battery, val_ratio=0.2
            )

            # 转换为tensor
            train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
            train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
            val_data_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)
            val_target_tensor = torch.tensor(val_target, dtype=torch.float32).to(device)
            test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
            test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

            # 创建数据加载器
            train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
            val_dataset = TensorDataset(val_data_tensor, val_target_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 初始化模型
            setup_seed(42)
            model = CollaborativeTransformerLSTM(
                input_size, hidden_size, output_size,
                num_layers=2, num_heads=num_heads, dropout=dropout
            ).to(device)

            # 优化器和损失函数
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15, verbose=False
            )

            # 早停
            early_stopping = ImprovedEarlyStopping(patience=50, min_delta=1e-6)

            # 训练循环
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                # 训练阶段
                model.train()
                epoch_train_loss = 0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    epoch_train_loss += loss.item()

                avg_train_loss = epoch_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # 验证阶段
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        epoch_val_loss += loss.item()

                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                # 学习率调度
                scheduler.step(avg_val_loss)

                # 监控梯度
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

                # 早停检查
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # 测试阶段
            model.eval()
            with torch.no_grad():
                test_pred = model(test_data_tensor)

            # 计算评估指标
            pred_np = test_pred.cpu().numpy()
            test_np = test_target_tensor.cpu().numpy()

            # 如果是多维输出，需要展平
            if pred_np.ndim > 1:
                pred_np = pred_np.flatten()
            if test_np.ndim > 1:
                test_np = test_np.flatten()

            mae, mse, rmse = evaluation(test_np, pred_np)
            print(f"Results for {test_battery}:")
            print(f"RMSE: {rmse*100:.3f}, MAE: {mae*100:.3f}")

            maes.append(mae)
            rmses.append(rmse)

        except Exception as e:
            print(f"Error processing {test_battery}: {e}")
            continue

    # 输出总体结果
    if len(maes) > 0 and len(rmses) > 0:
        print(f"\n=== Cross-validation results ===")
        print(f"Average RMSE: {np.mean(rmses):.6f}")
        print(f"Average MAE: {np.mean(maes):.6f}")
        print(f"RMSE std: {np.std(rmses):.6f}")
        print(f"MAE std: {np.std(maes):.6f}")
    else:
        print("No valid results obtained!")


if __name__ == "__main__":
    main()