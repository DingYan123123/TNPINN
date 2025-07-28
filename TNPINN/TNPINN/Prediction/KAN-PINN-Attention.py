import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import pywt  # 导入小波变换库
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
import warnings

warnings.filterwarnings('ignore')

# 原有参数设置
window_size = 10  # 设置窗口大小
forecast_steps = 10  # 预测步数
batch_size = 32  # 设置批次大小
input_size = 6  # 输入特征数
hidden_size = 64  # 隐藏层大小
output_size = forecast_steps  # 输出层大小
epochs = 1000  # 训练轮数
learningrate = 0.001  # 调整学习率适应新模型
weight_decay = 1e-5  # L2正则化强度


# 设置随机种子以确保可重复性
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ==================== Attention-KAN-PINN 模型组件 ====================

class BSpline(nn.Module):
    """B样条基函数实现"""

    def __init__(self, grid_size=8, spline_order=3):
        super(BSpline, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 创建B样条节点
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)

    def forward(self, x):
        """计算B样条基函数值"""
        # 确保输入在正确范围内
        x = torch.clamp(x, -0.99, 0.99)

        # 计算B样条基函数
        bases = []
        for i in range(self.grid_size + self.spline_order):
            basis = self.b_spline_basis(x, i, self.spline_order)
            bases.append(basis)

        return torch.stack(bases, dim=-1)

    def b_spline_basis(self, x, i, k):
        """递归计算B样条基函数"""
        if k == 0:
            return ((x >= self.grid[i]) & (x < self.grid[i + 1])).float()
        else:
            c1 = torch.zeros_like(x)
            c2 = torch.zeros_like(x)

            if self.grid[i + k] != self.grid[i]:
                c1 = (x - self.grid[i]) / (self.grid[i + k] - self.grid[i]) * \
                     self.b_spline_basis(x, i, k - 1)

            if self.grid[i + k + 1] != self.grid[i + 1]:
                c2 = (self.grid[i + k + 1] - x) / (self.grid[i + k + 1] - self.grid[i + 1]) * \
                     self.b_spline_basis(x, i + 1, k - 1)

            return c1 + c2


class KANLayer(nn.Module):
    """Kolmogorov-Arnold网络层"""

    def __init__(self, input_dim, output_dim, grid_size=8, spline_order=3):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        # B样条基函数
        self.bspline = BSpline(grid_size, spline_order)

        # 系数参数
        num_bases = grid_size + spline_order
        self.coefficients = nn.Parameter(
            torch.randn(output_dim, input_dim, num_bases) * 0.1
        )

        # 可选的残差连接权重
        self.residual_weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

    def forward(self, x):
        """前向传播"""
        batch_size = x.shape[0]

        # 计算B样条基函数
        basis_values = self.bspline(x)  # [batch_size, input_dim, num_basis]

        # 计算KAN输出
        output = torch.einsum('bin,oin->bo', basis_values, self.coefficients)

        # 添加残差连接
        residual = torch.sum(x.unsqueeze(1) * self.residual_weight.unsqueeze(0), dim=2)

        return output + residual


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 拼接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 输出投影
        output = self.W_o(attention_output)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_length=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PhysicsLoss(nn.Module):
    """物理约束损失函数"""

    def __init__(self, physics_weight=0.1):
        super(PhysicsLoss, self).__init__()
        self.physics_weight = physics_weight

    def forward(self, predictions, inputs):
        """计算物理约束损失"""
        physics_loss = torch.tensor(0.0, device=predictions.device)

        # 1. 单调性约束：电池容量应该单调递减
        if predictions.shape[1] > 1:
            capacity_diff = predictions[:, 1:] - predictions[:, :-1]
            monotonicity_loss = F.relu(capacity_diff).mean()  # 惩罚正的差值
            physics_loss += monotonicity_loss

        # 2. 边界约束：容量应该在合理范围内
        capacity_bound_loss = F.relu(predictions - 1.2).mean() + F.relu(-predictions + 0.1).mean()
        physics_loss += capacity_bound_loss

        # 3. 衰减率约束：衰减率应该在合理范围内
        if predictions.shape[1] > 2:
            decay_rate = torch.abs(predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2])
            decay_constraint = F.relu(decay_rate - 0.05).mean()  # 限制衰减率
            physics_loss += 0.5 * decay_constraint

        return self.physics_weight * physics_loss


class AttentionKANBlock(nn.Module):
    """注意力-KAN融合块"""

    def __init__(self, d_model, num_heads, kan_grid_size=8, dropout=0.1):
        super(AttentionKANBlock, self).__init__()

        # 多头注意力
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # KAN层
        self.kan_layer = KANLayer(d_model, d_model, kan_grid_size)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力子层
        attn_output, attention_weights = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # KAN子层
        if x.dim() == 3:  # [batch, seq, features]
            batch_size, seq_len, features = x.shape
            x_reshaped = x.view(-1, features)  # [batch*seq, features]
            kan_output = self.kan_layer(x_reshaped)
            kan_output = kan_output.view(batch_size, seq_len, -1)
        else:
            kan_output = self.kan_layer(x)

        x = self.norm2(x + self.dropout(kan_output))

        return x, attention_weights


class AttentionKANPINN(nn.Module):
    """Attention-KAN-PINN主模型"""

    def __init__(self,
                 input_size=6,
                 hidden_size=64,
                 output_size=10,
                 num_heads=8,
                 num_layers=3,
                 kan_grid_size=8,
                 dropout=0.1):
        super(AttentionKANPINN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 输入投影
        self.input_projection = nn.Linear(input_size, hidden_size)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_size, max_length=1000)

        # Attention-KAN块堆叠
        self.attention_kan_blocks = nn.ModuleList([
            AttentionKANBlock(hidden_size, num_heads, kan_grid_size, dropout)
            for _ in range(num_layers)
        ])

        # LSTM层（保持时序建模能力）
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # 物理损失
        self.physics_loss = PhysicsLoss(physics_weight=0.1)

        # 存储注意力权重
        self.attention_weights = []

    def forward(self, x):
        # 输入投影到模型维度
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_size]

        # 添加位置编码
        seq_len = x.shape[1]
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_size]

        # 通过Attention-KAN块
        self.attention_weights = []
        for block in self.attention_kan_blocks:
            x, attn_weights = block(x)
            self.attention_weights.append(attn_weights)

        # LSTM层
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        final_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # 输出投影
        output = self.output_projection(final_hidden)

        return output

    def compute_loss(self, predictions, targets, inputs):
        """计算总损失：数据损失 + 物理损失"""
        # 数据拟合损失
        data_loss = F.mse_loss(predictions, targets)

        # 物理约束损失
        physics_loss = self.physics_loss(predictions, inputs)

        # 总损失
        total_loss = data_loss + physics_loss

        return total_loss, data_loss, physics_loss


# ==================== 原有的辅助函数 ====================

def min_max_normalization(data):
    """Min-Max归一化方法，将数据缩放到 [0, 1] 范围内"""
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def z_score_standardization(data):
    """Z-score标准化方法，将数据转换为均值为0，标准差为1的分布"""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def custom_normalization(data):
    """归一化处理：将数据缩放到 [0, 1] 之间"""
    capacity_col = data[:, -1]
    max_capacity = np.max(capacity_col)
    SOH = capacity_col / max_capacity
    max_SOH = np.max(SOH)
    min_SOH = np.min(SOH)
    capacity_col_normalized = 2 * (SOH - min_SOH) / (max_SOH - min_SOH) - 1
    other_cols = data[:, :-1]
    min_vals = np.min(other_cols, axis=0)
    max_vals = np.max(other_cols, axis=0)
    other_cols_normalized = 2 * (other_cols - min_vals) / (max_vals - min_vals) - 1
    return np.column_stack((other_cols_normalized, capacity_col_normalized))


def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    """小波变换去噪函数（软阈值法）"""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


def build_sequences(text, window_size, forecast_steps):
    """生成时序序列"""
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]
        target = text[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)


def evaluation(y_test, y_predict):
    """评估函数"""
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, mse, rmse


class EarlyStopping:
    """早停类"""

    def __init__(self, patience=50, verbose=False, delta=0):
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
            self.best_model_wts = model.state_dict().copy()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict().copy()
            self.counter = 0
def min_max_normalization_per_feature(data):
    # 如果输入数据是 NumPy 数组，将其转换为 PyTorch 张量
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # 对每个特征（列）进行最小-最大归一化
    min_vals = torch.min(data, dim=0)[0]  # 每个特征的最小值，返回最小值和索引，这里我们取最小值
    max_vals = torch.max(data, dim=0)[0]  # 每个特征的最大值，返回最大值和索引，这里我们取最大值
    return (data - min_vals) / (max_vals - min_vals + 1e-6)  # 归一化公式，防止除以0的错误

# ==================== 主程序 ====================

# 数据加载和处理
folder_path = '../NASA Cleaned'  # 替换为电池文件所在的文件夹路径
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  # 获取所有CSV文件
battery_data = {}

# 对每个电池的数据进行处理
for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    # 提取后8列特征
    features = df.iloc[:, 1:].values  # 获取后8列特征
    last_column = df.iloc[:, -1].values # 获取最后一列特征
    # 应用小波变换去噪（软阈值处理）
    # last_column = np.apply_along_axis(wavelet_denoising, 0, last_column)
    features = np.column_stack((features[:,0:5], last_column/2))
    # 应用小波变换去噪（软阈值处理）
    features1 = np.apply_along_axis(wavelet_denoising, 0, features[:,0:5])  # 对每一列特征应用小波去噪
    features = np.column_stack((features1, last_column))
    features = min_max_normalization_per_feature(features)
    # 创建数据和目标
    data, target = build_sequences(features, window_size, forecast_steps)

    # 将每个电池的数据保存到字典中
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses = [], []
data_losses = []
physics_losses = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 训练过程
for test_battery, (test_data, test_target) in battery_data.items():
    print(f"Testing on battery: {test_battery}")

    # 训练集：所有电池数据，除了当前测试电池
    train_data = []
    train_target = []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)

    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    # 转换为PyTorch张量，并将其移动到GPU上（如果可用）
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    # 创建数据加载器（DataLoader）
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型参数
    setup_seed(0)

    # 使用Attention-KAN-PINN模型
    model = AttentionKANPINN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_heads=4,
        num_layers=2,
        kan_grid_size=8,
        dropout=0.1
    ).to(device)

    # 定义优化器（使用Adam优化器，适合复杂模型）
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 早停实例化
    early_stopping = EarlyStopping(patience=50, verbose=True)

    # 训练损失记录
    train_losses = []
    train_data_losses = []
    train_physics_losses = []

    print("开始训练Attention-KAN-PINN模型...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_physics_loss = 0
        batch_count = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            output = model(inputs)

            # 计算损失
            total_loss, data_loss, physics_loss = model.compute_loss(output, targets, inputs)

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_data_loss = epoch_data_loss / batch_count
            avg_physics_loss = epoch_physics_loss / batch_count

            train_losses.append(avg_loss)
            train_data_losses.append(avg_data_loss)
            train_physics_losses.append(avg_physics_loss)

            # 在验证集上计算损失
            model.eval()
            with torch.no_grad():
                val_pred = model(test_data_tensor)
                val_total_loss, val_data_loss, val_physics_loss = model.compute_loss(
                    val_pred, test_target_tensor, test_data_tensor
                )

            # 学习率调度
            scheduler.step(val_total_loss.item())

            # 使用早停
            early_stopping(val_total_loss.item(), model)
            if early_stopping.early_stop:
                print("Early stopping")
                model.load_state_dict(early_stopping.best_model_wts)
                break

            if epoch % 50 == 0:
                print(f'Epoch {epoch}: Total Loss={avg_loss:.6f} '
                      f'(Data={avg_data_loss:.6f}, Physics={avg_physics_loss:.6f}), '
                      f'Val Loss={val_total_loss.item():.6f}')

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        pred = model(test_data_tensor)

    # 评估模型
    pred_np = pred.detach().squeeze().cpu().numpy()
    test_np = test_target_tensor.detach().squeeze().cpu().numpy()

    # 确保维度正确
    if pred_np.ndim > 1:
        pred_np = pred_np.flatten()
    if test_np.ndim > 1:
        test_np = test_np.flatten()

    mae, mse, rmse = evaluation(test_np, pred_np)
    print(f"RMSE: {rmse * 100 :.3f}, MAE: {mae * 100 :.3f}")

    # 保存评估指标
    maes.append(mae)
    rmses.append(rmse)

    # 绘制预测与真实值的对比图
    plt.figure(figsize=(8, 8))
    plt.scatter(test_np, pred_np, s=100, color='dodgerblue', alpha=0.8)
    plt.plot([min(test_np), max(test_np)], [min(test_np), max(test_np)], 'r--', label='Ideal')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    # plt.title(f"Attention-KAN-PINN: {test_battery}")
    plt.legend()
    plt.grid(True)
    plt.show()

# 汇总交叉验证结果
print("\nAttention-KAN-PINN Cross-validation results:")
print(f"Average RMSE: {np.mean(rmses) * 100:.3f}")
print(f"Average MAE: {np.mean(maes) * 100:.3f}")

# 保存详细结果
results_df = pd.DataFrame({
    'Battery': list(battery_data.keys()),
    'RMSE': [rmse * 100 for rmse in rmses],
    'MAE': [mae * 100 for mae in maes]
})

print("\nDetailed Results by Battery:")
print(results_df)

# 绘制总结果对比图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(len(rmses)), [rmse * 100 for rmse in rmses], color='skyblue', alpha=0.7)
plt.xlabel('Battery Index')
plt.ylabel('RMSE (%)')
plt.title('RMSE by Battery (Attention-KAN-PINN)')
plt.xticks(range(len(rmses)), [f'B{i + 1}' for i in range(len(rmses))])
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(len(maes)), [mae * 100 for mae in maes], color='lightcoral', alpha=0.7)
plt.xlabel('Battery Index')
plt.ylabel('MAE (%)')
plt.title('MAE by Battery (Attention-KAN-PINN)')
plt.xticks(range(len(maes)), [f'B{i + 1}' for i in range(len(maes))])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 性能对比分析
print("\n" + "=" * 60)
print("Attention-KAN-PINN 模型特点和优势分析:")
print("=" * 60)
print("🎯 核心技术组件:")
print("   • Kolmogorov-Arnold网络 (KAN): 函数级别的可学习参数")
print("   • 多头注意力机制: 捕获序列间的长距离依赖关系")
print("   • 物理信息神经网络 (PINN): 集成电池衰减物理约束")
print("   • 位置编码: 保持时序信息的完整性")
print("\n🔬 物理约束优势:")
print("   • 单调性约束: 确保容量预测符合衰减趋势")
print("   • 边界约束: 限制预测值在物理合理范围内")
print("   • 衰减率约束: 控制容量变化速率的合理性")
print("\n⚡ 模型创新点:")
print("   • KAN替代传统MLP: 更强的函数拟合能力")
print("   • 注意力机制: 自适应关注重要时间步")
print("   • 物理损失: 提升预测的物理一致性")
print("   • 多层融合: 结合LSTM保持时序建模优势")

# 模型复杂度分析
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 模型规模:")
print(f"   • 总参数量: {total_params:,}")
print(f"   • 可训练参数: {trainable_params:,}")
print(f"   • 模型大小: ~{total_params * 4 / 1024 / 1024:.2f} MB")

# 性能提升建议
print(f"\n🚀 性能优化建议:")
print("   1. 调整KAN网格大小以平衡精度和计算效率")
print("   2. 尝试不同的物理约束权重组合")
print("   3. 使用更大的训练数据集提升泛化能力")
print("   4. 考虑集成多个模型进行预测融合")
print("   5. 针对特定电池类型微调超参数")

print("\n" + "=" * 60)
print("Attention-KAN-PINN 模型训练和评估完成！")
print("=" * 60)