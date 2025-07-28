import math
import os
import time
from math import sqrt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

# CEEMDAN相关导入
try:
    from PyEMD import CEEMDAN
    CEEMDAN_AVAILABLE = True
    print("CEEMDAN library imported successfully")
except ImportError:
    print("Warning: PyEMD not found. Please install it using: pip install EMD-signal")
    CEEMDAN_AVAILABLE = False

# 参数设置
window_size = 10
forecast_steps = 10
epochs = 1000
input_size = 6
embed_dim = 32
num_heads = 4
depth = 2
learningrate = 0.001
weight_decay = 1e-5
batch_size = 32
early_stopping_patience = 50

# CEEMDAN参数
use_ceemdan = True  # 是否使用CEEMDAN去噪
ceemdan_trials = 100  # CEEMDAN试验次数
noise_std = 0.005  # 噪声标准差
max_imf = -1  # 最大IMF数量 (-1表示自动)

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

# Transformer模型
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, forecast_steps, depth, max_len=5000, dropout=0):
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
        self.fc = nn.Linear(embed_dim, forecast_steps)

    def _get_positional_encoding(self, max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
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

# CEEMDAN去噪函数
def ceemdan_denoising(data, trials=100, noise_std=0.005, max_imf=-1, reconstruct_components=None):
    """
    使用CEEMDAN进行信号去噪
    
    Parameters:
    - data: 输入信号 (1D array)
    - trials: CEEMDAN试验次数
    - noise_std: 添加噪声的标准差
    - max_imf: 最大IMF数量，-1表示自动确定
    - reconstruct_components: 用于重构的IMF分量索引，None表示自动选择
    
    Returns:
    - denoised_signal: 去噪后的信号
    - imfs: 所有IMF分量
    """
    if not CEEMDAN_AVAILABLE:
        print("CEEMDAN not available, returning original data")
        return data, None
    
    try:
        # 初始化CEEMDAN对象
        ceemdan = CEEMDAN(trials=trials, noise_std=noise_std)
        if max_imf > 0:
            ceemdan.max_imf = max_imf
        
        # 执行CEEMDAN分解
        imfs = ceemdan(data)
        
        if reconstruct_components is None:
            # 自动选择重构分量：通常排除高频噪声分量（前1-2个IMF）
            # 这个策略可以根据具体数据调整
            if len(imfs) > 3:
                reconstruct_components = list(range(1, len(imfs)))  # 排除第一个高频分量
            else:
                reconstruct_components = list(range(len(imfs)))
        
        # 重构去噪信号
        denoised_signal = np.sum(imfs[reconstruct_components], axis=0)
        
        return denoised_signal, imfs
    
    except Exception as e:
        print(f"CEEMDAN denoising failed: {e}")
        return data, None


# 归一化函数
def min_max_normalization(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# 数据预处理（包含CEEMDAN去噪）
def preprocess_features(features, use_ceemdan=True):
    """
    对特征数据进行预处理，包括CEEMDAN去噪
    
    Parameters:
    - features: 原始特征数据
    - use_ceemdan: 是否使用CEEMDAN去噪
    
    Returns:
    - processed_features: 处理后的特征数据
    """
    processed_features = features.copy()
    
    if use_ceemdan and CEEMDAN_AVAILABLE:
        print("Applying CEEMDAN denoising to features...")
        
        # 对每个特征列进行CEEMDAN去噪
        for i in range(features.shape[1]):
            print(f"Denoising feature {i+1}/{features.shape[1]}...")
            
            # 应用CEEMDAN去噪
            denoised_feature, imfs = ceemdan_denoising(
                features[:, i], 
                trials=ceemdan_trials,
                noise_std=noise_std,
                max_imf=max_imf
            )
            
            if denoised_feature is not None:
                processed_features[:, i] = denoised_feature
            
            # 可选：显示IMF分量分析
            if imfs is not None and i == 0:  # 只为第一个特征显示分析
                plt.figure(figsize=(12, 8))
                plt.subplot(len(imfs)+2, 1, 1)
                plt.plot(features[:, i], 'b-', label='Original')
                plt.title(f'Original Signal - Feature {i+1}')
                plt.legend()
                
                for j, imf in enumerate(imfs):
                    plt.subplot(len(imfs)+2, 1, j+2)
                    plt.plot(imf, label=f'IMF {j+1}')
                    plt.title(f'IMF {j+1}')
                    plt.legend()
                
                plt.subplot(len(imfs)+2, 1, len(imfs)+2)
                plt.plot(features[:, i], 'b-', alpha=0.7, label='Original')
                plt.plot(denoised_feature, 'r-', label='CEEMDAN Denoised')
                plt.title('Comparison: Original vs Denoised')
                plt.legend()
                plt.tight_layout()
                plt.show()
    
    return processed_features

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

# 加载并处理数据
folder_path = '../NASA Cleaned'
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

print(f"Found {len(battery_files)} battery files")
print(f"CEEMDAN denoising: {'Enabled' if use_ceemdan else 'Disabled'}")

for i, file_name in enumerate(battery_files):
    print(f"\nProcessing file {i+1}/{len(battery_files)}: {file_name}")
    
    df = pd.read_csv(os.path.join(folder_path, file_name))
    features = df.iloc[:, 1:].values
    last_column = df.iloc[:, -1].values
    features = np.column_stack((features[:, 0:5], last_column / 2))
    
    # 应用CEEMDAN去噪
    if use_ceemdan:
        features = preprocess_features(features, use_ceemdan=True)
    
    # 归一化
    features = min_max_normalization(features)
    
    # 构建序列
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)
    
    print(f"Generated {len(data)} sequences for {file_name}")

# 模型训练和测试
maes, rmses, times = [], [], []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')

for fold_idx, (test_battery, (test_data, test_target)) in enumerate(battery_data.items()):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{len(battery_data)}: Testing on battery {test_battery}")
    print(f"{'='*60}")
    
    start_time = time.time()

    # 准备训练数据
    train_data, train_target = [], []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)
    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    # 转换为张量
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(train_data_tensor, train_target_tensor),
                              batch_size=batch_size, shuffle=True)

    # 初始化模型
    setup_seed(0)
    model = TransformerTimeSeriesModel(input_size, embed_dim, num_heads, forecast_steps, depth).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    train_losses = []
    train_start_time = time.time()

    # 训练循环
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

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(test_data_tensor)
            val_loss = criterion(val_pred, test_target_tensor).item()
        
        early_stopping(val_loss, model)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    train_end_time = time.time()
    training_time = train_end_time - train_start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # 使用最佳模型进行预测
    model.load_state_dict(early_stopping.best_model_wts)
    model.eval()
    with torch.no_grad():
        pred = model(test_data_tensor)

    # 评估性能
    pred_np = pred.cpu().numpy()
    true_np = test_target_tensor.cpu().numpy()
    mae, mse, rmse = evaluation(true_np, pred_np)
    print(f"Results for {test_battery}:")
    print(f"  RMSE: {rmse * 100:.3f}%")
    print(f"  MAE: {mae * 100:.3f}%")
    print(f"  Training Time: {training_time:.2f}s")
    
    maes.append(mae * 100)
    rmses.append(rmse * 100)
    times.append(training_time)

    # 可视化训练损失
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.title(f"Training Loss for {test_battery}" + (" (with CEEMDAN)" if use_ceemdan else ""))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 可视化预测 vs 真实值
    plt.figure(figsize=(10, 8))
    plt.scatter(true_np.flatten(), pred_np.flatten(), color='dodgerblue', s=60, alpha=0.7, edgecolors='navy', linewidth=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction', linewidth=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Prediction vs True for {test_battery}" + (" (with CEEMDAN)" if use_ceemdan else ""))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加R²相关系数
    correlation = np.corrcoef(true_np.flatten(), pred_np.flatten())[0, 1]
    plt.text(0.05, 0.95, f'R² = {correlation**2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.show()

# 最终汇总结果
print(f"\n{'='*60}")
print("CROSS-VALIDATION RESULTS SUMMARY")
print(f"{'='*60}")
print(f"CEEMDAN Denoising: {'Enabled' if use_ceemdan else 'Disabled'}")
print(f"Number of folds: {len(battery_files)}")
print(f"\nPerformance Metrics:")
print(f"  Average RMSE: {np.mean(rmses):.3f}% (±{np.std(rmses):.3f})")
print(f"  Average MAE:  {np.mean(maes):.3f}% (±{np.std(maes):.3f})")
print(f"  Average Training Time: {np.mean(times):.2f}s (±{np.std(times):.2f})")

print(f"\nDetailed Results by Battery:")
for i, (battery, rmse, mae, time_taken) in enumerate(zip(battery_files, rmses, maes, times)):
    print(f"  {i+1:2d}. {battery:25s}: RMSE={rmse:6.3f}%, MAE={mae:6.3f}%, Time={time_taken:6.2f}s")
