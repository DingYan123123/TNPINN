import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pywt  # 导入小波变换库
from PyEMD import EEMD, CEEMDAN
import warnings

warnings.filterwarnings('ignore')

# 原有参数设置
window_size = 20  # 设置窗口大小
forecast_steps = 10  # 预测步数
batch_size = 32  # 设置批次大小
input_size = 6  # 输入特征数
hidden_size = 64  # 隐藏层大小
output_size = forecast_steps  # 输出层大小
epochs = 1000  # 训练轮数
learningrate = 0.001 # 学习率
weight_decay = 0  # L2正则化强度，可调

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

# ==================== Ist-SBiLSTM模型组件 ====================

class ICEEMDAN:
    """改进的完整集成经验模态分解"""

    def __init__(self, trials=50, noise_scale=0.005, ext_EMD=None):
        self.trials = trials
        self.noise_scale = noise_scale
        self.ext_EMD = ext_EMD or EEMD()

    def iceemdan(self, S, max_imf=-1):
        """ICEEMDAN分解算法"""
        try:
            # 使用CEEMDAN作为ICEEMDAN的近似实现
            ceemdan = CEEMDAN(trials=self.trials, noise_scale=self.noise_scale)
            IMFs = ceemdan.ceemdan(S, max_imf=max_imf)

            # 分离IMF和残余分量
            imfs = IMFs[:-1]  # 所有IMF分量
            res = IMFs[-1]  # 残余分量

            return np.array(imfs), res
        except:
            # 如果CEEMDAN失败，使用简单的EMD分解
            emd = EEMD(trials=self.trials, noise_width=self.noise_scale)
            IMFs = emd.eemd(S, max_imf=max_imf)
            imfs = IMFs[:-1]
            res = IMFs[-1]
            return np.array(imfs), res

class SparseAttention(nn.Module):
    """稀疏注意力机制"""

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(SparseAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 生成Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # 稀疏化处理：只保留top-k个注意力权重
        top_k = min(seq_len // 2, 10)  # 稀疏度控制
        if top_k > 0:
            _, top_indices = torch.topk(scores, top_k, dim=-1)

            # 创建稀疏掩码
            sparse_mask = torch.zeros_like(scores)
            sparse_mask.scatter_(-1, top_indices, 1)
            scores = scores * sparse_mask + (1 - sparse_mask) * (-1e9)

        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return context, attention_weights

class SBiLSTM(nn.Module):
    """具有稀疏注意力机制的双向LSTM"""

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, output_size=1):
        super(SBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # 稀疏注意力机制
        self.sparse_attention = SparseAttention(hidden_size * 2, num_heads=4, dropout=dropout)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # BiLSTM前向传播
        lstm_out, _ = self.bilstm(x)

        # 应用稀疏注意力机制
        attended_out, attention_weights = self.sparse_attention(lstm_out)

        # 取最后一个时间步的输出
        final_output = attended_out[:, -1, :]

        # 通过全连接层得到最终预测
        output = self.fc(final_output)

        return output

class GravitationalSearchAlgorithm:
    """引力搜索算法用于超参数优化"""

    def __init__(self, n_agents=10, max_iter=20, bounds=None):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.bounds = bounds or [0, 1]
        self.G0 = 100  # 初始引力常数
        self.alpha = 20  # 引力常数衰减参数

    def initialize_population(self, dim):
        """初始化种群"""
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_agents, dim))

    def calculate_fitness(self, population, fitness_func):
        """计算适应度"""
        fitness = []
        for agent in population:
            try:
                fit = fitness_func(agent)
                fitness.append(fit if not np.isnan(fit) else 1e6)
            except:
                fitness.append(1e6)
        return np.array(fitness)

    def update_G(self, iteration):
        """更新引力常数"""
        return self.G0 * np.exp(-self.alpha * iteration / self.max_iter)

    def optimize(self, fitness_func, dim):
        """GSA优化主循环"""
        # 初始化种群
        population = self.initialize_population(dim)
        best_fitness = float('inf')
        best_agent = None

        for iteration in range(self.max_iter):
            # 计算适应度
            fitness = self.calculate_fitness(population, fitness_func)

            # 更新最佳个体
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_agent = population[current_best_idx].copy()

            # 计算质量和引力
            worst_fitness = np.max(fitness)
            best_fitness_iter = np.min(fitness)

            # 避免除零错误
            if worst_fitness == best_fitness_iter:
                masses = np.ones(self.n_agents)
            else:
                masses = (fitness - worst_fitness) / (best_fitness_iter - worst_fitness)

            # 更新引力常数
            G = self.update_G(iteration)

            # 计算加速度和速度
            accelerations = np.zeros_like(population)

            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j:
                        r = np.linalg.norm(population[i] - population[j]) + 1e-10
                        force = G * masses[j] * (population[j] - population[i]) / r
                        accelerations[i] += force

            # 更新位置
            velocities = np.random.rand(*population.shape) * accelerations
            population = population + velocities

            # 边界处理
            population = np.clip(population, self.bounds[0], self.bounds[1])

        return best_agent, best_fitness

class GStackingEnsemble:
    """GSA优化的堆叠集成模型"""

    def __init__(self, base_models=None, meta_model=None):
        if base_models is None:
            self.base_models = [
                RandomForestRegressor(n_estimators=50, random_state=42),
                GradientBoostingRegressor(n_estimators=50, random_state=42),
                SVR(kernel='rbf', C=1.0),
            ]
        else:
            self.base_models = base_models

        self.meta_model = meta_model or LinearRegression()
        self.gsa = GravitationalSearchAlgorithm(n_agents=10, max_iter=15)
        self.model_weights = None

    def fitness_function(self, weights):
        """GSA的适应度函数"""
        try:
            # 使用权重组合基模型预测
            weighted_pred = np.zeros(len(self.base_predictions[0]))
            weights = weights / (np.sum(weights) + 1e-10)  # 归一化权重

            for i, pred in enumerate(self.base_predictions):
                if i < len(weights):
                    weighted_pred += weights[i] * pred

            # 计算RMSE作为适应度（越小越好）
            mse = mean_squared_error(self.y_val, weighted_pred)
            return np.sqrt(mse)
        except:
            return 1e6  # 返回大值表示差的适应度

    def fit(self, X_train, y_train, X_val, y_val):
        """训练堆叠集成模型"""
        # 训练基模型
        self.base_predictions = []
        self.y_val = y_val

        for model in self.base_models:
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                self.base_predictions.append(pred)
            except:
                # 如果模型训练失败，添加零预测
                self.base_predictions.append(np.zeros(len(y_val)))

        # 使用GSA优化基模型权重
        if len(self.base_predictions) > 0:
            best_weights, _ = self.gsa.optimize(
                self.fitness_function,
                len(self.base_models)
            )
            self.model_weights = best_weights / (np.sum(best_weights) + 1e-10)
        else:
            self.model_weights = np.ones(len(self.base_models)) / len(self.base_models)

        # 训练元模型
        try:
            meta_features = np.column_stack(self.base_predictions)
            self.meta_model.fit(meta_features, y_val)
        except:
            pass

    def predict(self, X_test):
        """预测"""
        # 获取基模型预测
        base_preds = []
        for model in self.base_models:
            try:
                pred = model.predict(X_test)
                base_preds.append(pred)
            except:
                base_preds.append(np.zeros(len(X_test)))

        # 使用优化的权重组合
        weighted_pred = np.zeros(len(base_preds[0]))
        for i, pred in enumerate(base_preds):
            weighted_pred += self.model_weights[i] * pred

        return weighted_pred

class IstSBiLSTMModel(nn.Module):
    """集成的Ist-SBiLSTM模型"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(IstSBiLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 用于处理RES分量的SBiLSTM
        self.res_model = SBiLSTM(
            input_size=1,  # RES分量是一维的
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )

        # 用于处理IMF分量的简化LSTM
        self.imf_model = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size // 4,
            num_layers=1,
            batch_first=True
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size // 4 + output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        # ICEEMDAN分解器
        self.iceemdan = ICEEMDAN(trials=20, noise_scale=0.005)

    def linear_regression_detection(self, data, threshold=2.0):
        """线性回归异常检测和修正"""
        try:
            x = np.arange(len(data)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x, data)

            # 预测值和残差
            pred = lr.predict(x)
            residuals = np.abs(data - pred)

            # 异常检测
            std_residual = np.std(residuals)
            anomaly_mask = residuals > threshold * std_residual

            # 数据修正
            corrected_data = data.copy()
            corrected_data[anomaly_mask] = pred[anomaly_mask]

            return corrected_data
        except:
            return data

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # 提取最后一列特征（容量特征）进行分解
        capacity_data = x[:, :, -1]  # [batch_size, seq_len]

        predictions = []

        for batch_idx in range(batch_size):
            single_capacity = capacity_data[batch_idx].detach().cpu().numpy()

            # 1. 异常检测和修正
            corrected_data = self.linear_regression_detection(single_capacity)

            # 2. ICEEMDAN分解
            try:
                imfs, res = self.iceemdan.iceemdan(corrected_data)

                # 3. 处理RES分量
                res_input = torch.FloatTensor(res).unsqueeze(0).unsqueeze(-1).to(x.device)
                if res_input.shape[1] >= seq_len:
                    res_input = res_input[:, :seq_len, :]
                else:
                    # 如果RES长度不够，进行填充
                    padding = torch.zeros(1, seq_len - res_input.shape[1], 1).to(x.device)
                    res_input = torch.cat([res_input, padding], dim=1)

                res_pred = self.res_model(res_input)

                # 4. 处理IMF分量（简化处理，只用第一个IMF）
                if len(imfs) > 0:
                    imf_data = imfs[0]
                    imf_input = torch.FloatTensor(imf_data).unsqueeze(0).unsqueeze(-1).to(x.device)
                    if imf_input.shape[1] >= seq_len:
                        imf_input = imf_input[:, :seq_len, :]
                    else:
                        padding = torch.zeros(1, seq_len - imf_input.shape[1], 1).to(x.device)
                        imf_input = torch.cat([imf_input, padding], dim=1)

                    imf_out, _ = self.imf_model(imf_input)
                    imf_feature = imf_out[:, -1, :]  # 取最后一个时间步
                else:
                    imf_feature = torch.zeros(1, self.hidden_size // 4).to(x.device)

                # 5. 融合预测
                combined_features = torch.cat([imf_feature, res_pred], dim=-1)
                final_pred = self.fusion_layer(combined_features)

                predictions.append(final_pred)

            except Exception as e:
                # 如果分解失败，使用原始数据
                original_input = x[batch_idx:batch_idx + 1, :, -1:]
                res_pred = self.res_model(original_input)
                predictions.append(res_pred)

        # 合并所有批次的预测
        output = torch.cat(predictions, dim=0)
        return output

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
    # print(battery_data)

# 交叉验证
maes, rmses = [], []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建PDF文件来保存所有图像
pdf_path = 'prediction_results.pdf'
pdf = PdfPages(pdf_path)

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

    # 使用Ist-SBiLSTM模型替代原有的WindowedLSTM
    model = IstSBiLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=2,
        dropout=0.05
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    # 早停实例化
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # 训练损失记录
    train_losses = []

    print("开始训练Ist-SBiLSTM模型...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for inputs, targets in train_loader:
            try:
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)

                # 前向传播
                output = model(inputs)
                loss = criterion(output, targets)  # 计算损失

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            except Exception as e:
                print(f"Batch error: {e}")
                continue

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)

            # 在验证集上计算损失
            model.eval()
            with torch.no_grad():
                try:
                    val_pred = model(test_data_tensor)
                    val_loss = criterion(val_pred, test_target_tensor)

                    # 使用早停
                    early_stopping(val_loss.item(), model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        model.load_state_dict(early_stopping.best_model_wts)
                        break

                except Exception as e:
                    print(f"Validation error: {e}")
                    continue

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Training Loss: {avg_loss:.6f}')

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        try:
            pred = model(test_data_tensor)
        except Exception as e:
            print(f"Prediction error: {e}")
            # 如果预测失败，使用零预测
            pred = torch.zeros_like(test_target_tensor)

    # 评估模型
    pred_np = pred.detach().squeeze().cpu().numpy()
    test_np = test_target_tensor.detach().squeeze().cpu().numpy()

    # 确保维度匹配
    if pred_np.ndim > 1:
        pred_np = pred_np.flatten()
    if test_np.ndim > 1:
        test_np = test_np.flatten()

    mae, mse, rmse = evaluation(test_np, pred_np)
    print(f"RMSE: {rmse * 100 :.3f}, MAE: {mae * 100 :.3f}")

    # 保存评估指标
    maes.append(mae)
    rmses.append(rmse)

    # 绘制预测与真实值的对比图并保存为以电池名命名的PDF
    battery_pdf_path = f"{test_battery}_prediction.pdf"
    with PdfPages(battery_pdf_path) as battery_pdf:
        plt.figure(figsize=(8, 8))
        plt.scatter(test_np, pred_np, s=100, color='dodgerblue', alpha=0.8)
        plt.plot([min(test_np), max(test_np)], [min(test_np), max(test_np)], 'r--', label='Ideal')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        # plt.title(f"Ist-SBiLSTM Prediction Results for {test_battery}")
        plt.legend()
        plt.grid(True)
        battery_pdf.savefig()  # 保存到以电池命名的PDF
        plt.close()
    print(f"Plot saved to {battery_pdf_path}")

# 汇总交叉验证结果
print("\nIst-SBiLSTM Cross-validation results:")
print(f"Average RMSE: {np.mean(rmses) * 100:.3f}, Average MAE: {np.mean(maes) * 100:.3f}")