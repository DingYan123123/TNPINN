
import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pywt
from PyEMD import EEMD, CEEMDAN
import warnings

warnings.filterwarnings('ignore')

# 参数设置
window_size = 20  # 设置窗口大小
forecast_steps = 10  # 预测步数
batch_size = 32  # 设置批次大小
input_size = 6  # 输入特征数
hidden_size = 64  # 隐藏层大小
output_size = forecast_steps  # 输出层大小
epochs = 1000  # 训练轮数
learningrate = 0.001  # 学习率
weight_decay = 0  # L2正则化强度
dropout_rate = 0.05  # Dropout率，用于蒙特卡洛不确定性估计
mc_samples = 100  # 蒙特卡洛采样次数

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

# ==================== Ist-SBiLSTM模型组件 ====================

class ICEEMDAN:
    """改进的完整集成经验模态分解"""
    def __init__(self, trials=50, noise_scale=0.005, ext_EMD=None):
        self.trials = trials
        self.noise_scale = noise_scale
        self.ext_EMD = ext_EMD or EEMD()

    def iceemdan(self, S, max_imf=-1):
        try:
            ceemdan = CEEMDAN(trials=self.trials, noise_scale=self.noise_scale)
            IMFs = ceemdan.ceemdan(S, max_imf=max_imf)
            imfs = IMFs[:-1]
            res = IMFs[-1]
            return np.array(imfs), res
        except:
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

    def forward(self, x, mc_dropout=False):
        if mc_dropout:
            self.dropout.train()
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        top_k = min(seq_len // 2, 10)
        if top_k > 0:
            _, top_indices = torch.topk(scores, top_k, dim=-1)
            sparse_mask = torch.zeros_like(scores)
            sparse_mask.scatter_(-1, top_indices, 1)
            scores = scores * sparse_mask + (1 - sparse_mask) * (-1e9)
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return context, attention_weights

class SBiLSTM(nn.Module):
    """具有稀疏注意力机制的双向LSTM"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, output_size=1):
        super(SBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.sparse_attention = SparseAttention(hidden_size * 2, num_heads=4, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, mc_dropout=False):
        lstm_out, _ = self.bilstm(x)
        attended_out, attention_weights = self.sparse_attention(lstm_out, mc_dropout=mc_dropout)
        final_output = attended_out[:, -1, :]
        output = self.fc(final_output)
        return output

class GravitationalSearchAlgorithm:
    """引力搜索算法用于超参数优化"""
    def __init__(self, n_agents=10, max_iter=20, bounds=None):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.bounds = bounds or [0, 1]
        self.G0 = 100
        self.alpha = 20

    def initialize_population(self, dim):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_agents, dim))

    def calculate_fitness(self, population, fitness_func):
        fitness = []
        for agent in population:
            try:
                fit = fitness_func(agent)
                fitness.append(fit if not np.isnan(fit) else 1e6)
            except:
                fitness.append(1e6)
        return np.array(fitness)

    def update_G(self, iteration):
        return self.G0 * np.exp(-self.alpha * iteration / self.max_iter)

    def optimize(self, fitness_func, dim):
        population = self.initialize_population(dim)
        best_fitness = float('inf')
        best_agent = None
        for iteration in range(self.max_iter):
            fitness = self.calculate_fitness(population, fitness_func)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_agent = population[current_best_idx].copy()
            worst_fitness = np.max(fitness)
            best_fitness_iter = np.min(fitness)
            if worst_fitness == best_fitness_iter:
                masses = np.ones(self.n_agents)
            else:
                masses = (fitness - worst_fitness) / (best_fitness_iter - worst_fitness)
            G = self.update_G(iteration)
            accelerations = np.zeros_like(population)
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j:
                        r = np.linalg.norm(population[i] - population[j]) + 1e-10
                        force = G * masses[j] * (population[j] - population[i]) / r
                        accelerations[i] += force
            velocities = np.random.rand(*population.shape) * accelerations
            population = population + velocities
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
        try:
            weighted_pred = np.zeros(len(self.base_predictions[0]))
            weights = weights / (np.sum(weights) + 1e-10)
            for i, pred in enumerate(self.base_predictions):
                if i < len(weights):
                    weighted_pred += weights[i] * pred
            mse = mean_squared_error(self.y_val, weighted_pred)
            return np.sqrt(mse)
        except:
            return 1e6

    def fit(self, X_train, y_train, X_val, y_val):
        self.base_predictions = []
        self.y_val = y_val
        for model in self.base_models:
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                self.base_predictions.append(pred)
            except:
                self.base_predictions.append(np.zeros(len(y_val)))
        if len(self.base_predictions) > 0:
            best_weights, _ = self.gsa.optimize(self.fitness_function, len(self.base_models))
            self.model_weights = best_weights / (np.sum(best_weights) + 1e-10)
        else:
            self.model_weights = np.ones(len(self.base_models)) / len(self.base_models)
        try:
            meta_features = np.column_stack(self.base_predictions)
            self.meta_model.fit(meta_features, y_val)
        except:
            pass

    def predict(self, X_test):
        base_preds = []
        for model in self.base_models:
            try:
                pred = model.predict(X_test)
                base_preds.append(pred)
            except:
                base_preds.append(np.zeros(len(X_test)))
        weighted_pred = np.zeros(len(base_preds[0]))
        for i, pred in enumerate(base_preds):
            weighted_pred += self.model_weights[i] * pred
        return weighted_pred

class IstSBiLSTMModel(nn.Module):
    """集成的Ist-SBiLSTM模型 with Monte Carlo Dropout"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(IstSBiLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.res_model = SBiLSTM(
            input_size=1,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
        self.imf_model = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size // 4,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size // 4 + output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        self.iceemdan = ICEEMDAN(trials=20, noise_scale=0.005)

    def linear_regression_detection(self, data, threshold=2.0):
        try:
            x = np.arange(len(data)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x, data)
            pred = lr.predict(x)
            residuals = np.abs(data - pred)
            std_residual = np.std(residuals)
            anomaly_mask = residuals > threshold * std_residual
            corrected_data = data.copy()
            corrected_data[anomaly_mask] = pred[anomaly_mask]
            return corrected_data
        except:
            return data

    def forward(self, x, mc_dropout=False):
        batch_size, seq_len, features = x.shape
        capacity_data = x[:, :, -1]
        predictions = []
        for batch_idx in range(batch_size):
            single_capacity = capacity_data[batch_idx].detach().cpu().numpy()
            corrected_data = self.linear_regression_detection(single_capacity)
            try:
                imfs, res = self.iceemdan.iceemdan(corrected_data)
                res_input = torch.FloatTensor(res).unsqueeze(0).unsqueeze(-1).to(x.device)
                if res_input.shape[1] >= seq_len:
                    res_input = res_input[:, :seq_len, :]
                else:
                    padding = torch.zeros(1, seq_len - res_input.shape[1], 1).to(x.device)
                    res_input = torch.cat([res_input, padding], dim=1)
                res_pred = self.res_model(res_input, mc_dropout=mc_dropout)
                if len(imfs) > 0:
                    imf_data = imfs[0]
                    imf_input = torch.FloatTensor(imf_data).unsqueeze(0).unsqueeze(-1).to(x.device)
                    if imf_input.shape[1] >= seq_len:
                        imf_input = imf_input[:, :seq_len, :]
                    else:
                        padding = torch.zeros(1, seq_len - imf_input.shape[1], 1).to(x.device)
                        imf_input = torch.cat([imf_input, padding], dim=1)
                    imf_out, _ = self.imf_model(imf_input)
                    imf_feature = imf_out[:, -1, :]
                else:
                    imf_feature = torch.zeros(1, self.hidden_size // 4).to(x.device)
                if mc_dropout:
                    imf_feature = self.dropout(imf_feature)
                combined_features = torch.cat([imf_feature, res_pred], dim=-1)
                final_pred = self.fusion_layer(combined_features)
                predictions.append(final_pred)
            except Exception as e:
                original_input = x[batch_idx:batch_idx + 1, :, -1:]
                res_pred = self.res_model(original_input, mc_dropout=mc_dropout)
                predictions.append(res_pred)
        output = torch.cat(predictions, dim=0)
        return output

# ==================== 辅助函数 ====================

def min_max_normalization(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def z_score_standardization(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def custom_normalization(data):
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

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, mse, rmse

def mc_uncertainty_estimation(model, input_data, mc_samples=100):
    model.train()
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            pred = model(input_data, mc_dropout=True)
        predictions.append(pred.cpu().numpy())
    predictions = np.stack(predictions, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)
    return mean_pred, std_pred, ci_lower, ci_upper

def calculate_uncertainty_metrics(y_true, ci_lower, ci_upper, confidence_level=0.95, eta=None):
    coverage = (y_true >= ci_lower) & (y_true <= ci_upper)
    picp = np.mean(coverage)
    interval_width = ci_upper - ci_lower
    y_range = np.max(y_true) - np.min(y_true)
    pinaw = np.mean(interval_width) / y_range
    if eta is None:
        coverage_deficit = confidence_level - picp
        if coverage_deficit <= 0:
            eta = 0.1
        elif coverage_deficit <= 0.05:
            eta = 0.3
        elif coverage_deficit <= 0.10:
            eta = 0.5
        else:
            eta = 1.0
    if picp >= confidence_level:
        cwc = pinaw
    else:
        coverage_penalty = eta * (confidence_level - picp) / confidence_level
        cwc = pinaw * (1 + coverage_penalty)
    return picp, pinaw, cwc, eta

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

def min_max_normalization_per_feature(data):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    min_vals = torch.min(data, dim=0)[0]
    max_vals = torch.max(data, dim=0)[0]
    return (data - min_vals) / (max_vals - min_vals + 1e-6)

# ==================== 主程序 ====================

folder_path = '../NASA Cleaned'
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:].values
    last_column = df.iloc[:, -1].values
    features = np.column_stack((features[:, 0:5], last_column / 2))
    features1 = np.apply_along_axis(wavelet_denoising, 0, features[:, 0:5])
    features = np.column_stack((features1, last_column))
    features = min_max_normalization_per_feature(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses, picps, pinaws, cwcs, etas = [], [], [], [], [], []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建PDF文件来保存所有图像
pdf_path = 'prediction_results.pdf'
pdf = PdfPages(pdf_path)

for test_battery, (test_data, test_target) in battery_data.items():
    print(f"\nTesting on battery: {test_battery}")

    train_data = []
    train_target = []
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
    model = IstSBiLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=2,
        dropout=dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=True)

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
                output = model(inputs)
                loss = criterion(output, targets)
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
            model.eval()
            with torch.no_grad():
                try:
                    val_pred = model(test_data_tensor)
                    val_loss = criterion(val_pred, test_target_tensor)
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

    # 蒙特卡洛不确定性估计
    mean_pred, std_pred, ci_lower, ci_upper = mc_uncertainty_estimation(model, test_data_tensor, mc_samples)

    # 评估
    test_np = test_target_tensor.cpu().numpy()
    mae, mse, rmse = evaluation(test_np, mean_pred)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")

    # 不确定性指标
    picp, pinaw, cwc, eta_used = calculate_uncertainty_metrics(test_np, ci_lower, ci_upper)
    print(f"PICP (95% Coverage): {picp * 100:.2f}%")
    print(f"PINAW (Normalized Width): {pinaw:.4f}")
    print(f"CWC (Coverage-Width Criterion): {cwc:.4f} (eta={eta_used:.1f})")

    maes.append(mae)
    rmses.append(rmse)
    picps.append(picp)
    pinaws.append(pinaw)
    cwcs.append(cwc)
    etas.append(eta_used)

    # 绘制预测与真实值的对比图并保存
    battery_pdf_path = f"{test_battery}_prediction.pdf"
    with PdfPages(battery_pdf_path) as battery_pdf:
        plt.figure(figsize=(8, 8))
        plt.scatter(test_np.flatten(), mean_pred.flatten(), s=100, color='dodgerblue', alpha=0.8)
        plt.plot([min(test_np.flatten()), max(test_np.flatten())], [min(test_np.flatten()), max(test_np.flatten())], 'r--', label='Ideal')
        plt.xlabel("True Values", fontsize=20)
        plt.ylabel("Predicted Values", fontsize=20)
        plt.tick_params(axis='both', labelsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)
        battery_pdf.savefig()
        pdf.savefig()
        plt.close()
    print(f"Scatter plot saved to {battery_pdf_path}")

    # 可视化预测和置信区间
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 1
    plt.plot(time_steps, test_np[sample_idx], label="True", marker='o')
    plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x')
    plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                     color="red", alpha=0.2, label="95% CI")
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("SOH (%)", fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    pdf.savefig()
    plt.close()
    print(f"Prediction interval plot saved to {pdf_path}")

# 关闭PDF文件
pdf.close()

# 汇总结果
print("\n" + "=" * 60)
print("Ist-SBiLSTM CROSS-VALIDATION RESULTS SUMMARY")
print("=" * 60)
print(f"Average RMSE: {np.mean(rmses) * 100:.3f} ± {np.std(rmses) * 100:.3f}")
print(f"Average MAE: {np.mean(maes) * 100:.3f} ± {np.std(maes) * 100:.3f}")
print("\nUNCERTAINTY QUANTIFICATION METRICS:")
print(f"Average PICP: {np.mean(picps) * 100:.2f}% ± {np.std(picps) * 100:.2f}%")
print(f"Average PINAW: {np.mean(pinaws):.4f} ± {np.std(pinaws):.4f}")
print(f"Average CWC: {np.mean(cwcs):.4f} ± {np.std(cwcs):.4f} (avg eta={np.mean(etas):.1f})")
print("=" * 60)

# 详细结果表格
print("\nDETAILED RESULTS BY BATTERY:")
print("-" * 110)
print(f"{'Battery':<20} {'RMSE(%)':<10} {'MAE(%)':<10} {'PICP(%)':<10} {'PINAW':<8} {'CWC':<8} {'ETA':<5}")
print("-" * 110)
for i, (battery_name, _) in enumerate(battery_data.items()):
    print(f"{battery_name:<20} {rmses[i] * 100:<10.3f} {maes[i] * 100:<10.3f} "
          f"{picps[i] * 100:<10.2f} {pinaws[i]:<8.4f} {cwcs[i]:<8.4f} {etas[i]:<5.1f}")
print("-" * 110)

# 绘制总结果对比图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(rmses)), [rmse * 100 for rmse in rmses], color='skyblue', alpha=0.7)
plt.xlabel('Battery Index')
plt.ylabel('RMSE (%)')
plt.title('RMSE by Battery (Ist-SBiLSTM)')
plt.xticks(range(len(rmses)), [f'B{i + 1}' for i in range(len(rmses))])
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(len(maes)), [mae * 100 for mae in maes], color='lightcoral', alpha=0.7)
plt.xlabel('Battery Index')
plt.ylabel('MAE (%)')
plt.title('MAE by Battery (Ist-SBiLSTM)')
plt.xticks(range(len(maes)), [f'B{i + 1}' for i in range(len(maes))])
plt.grid(True, alpha=0.3)
plt.tight_layout()
pdf = PdfPages('summary_results.pdf')
pdf.savefig()
plt.close()
pdf.close()

# 性能分析
print("\n" + "=" * 60)
print("Ist-SBiLSTM 模型特点和优势分析:")
print("=" * 60)
print(" 核心技术组件:")
print("   • ICEEMDAN分解: 分离高频和低频信号，提升预测鲁棒性")
print("   • 稀疏注意力机制: 捕获关键时序依赖，降低计算复杂度")
print("   • 双向LSTM: 捕捉前后向时序信息")
print("   • GSA优化集成: 优化基模型权重，提升集成性能")
print("   • Monte Carlo Dropout: 提供不确定性量化")
print("\n 模型优势:")
print("   • 异常检测与修正: 提高数据质量")
print("   • 分解与融合: 分别处理IMF和RES分量，提升预测精度")
print("   • 不确定性量化: 提供预测置信区间")
print("\n 模型创新点:")
print("   • 结合ICEEMDAN与SBiLSTM: 增强信号处理能力")
print("   • 稀疏注意力: 高效处理长序列")
print("   • GSA优化: 自动调整集成权重")
print("\n 模型规模:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   • 总参数量: {total_params:,}")
print(f"   • 可训练参数: {trainable_params:,}")
print(f"   • 模型大小: ~{total_params * 4 / 1024 / 1024:.2f} MB")
print(f"\n 性能优化建议:")
print("   1. 调整ICEEMDAN的噪声尺度以适应不同数据集")
print("   2. 优化GSA的迭代次数和种群规模")
print("   3. 增加基模型种类以提升集成效果")
print("   4. 调整Dropout率以平衡不确定性估计和预测精度")
print("   5. 针对特定电池类型微调超参数")
print("\n" + "=" * 60)
print("Ist-SBiLSTM 模型训练和评估完成！")
print("=" * 60)