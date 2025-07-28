import pandas as pd

# 加载上传的CSV文件
file_path = '../NASA/B0018charge.csv'
data = pd.read_csv(file_path)

# 设定恒流（CC）和恒压（CV）模式的阈值
cc_current_threshold = 1.5  # A (恒流模式电流阈值)
cv_voltage_threshold = 4.2  # V (恒压模式电压阈值)
cv_current_threshold = 0.02  # A (20mA，恒压模式电流阈值)

# 初始化一个空列表来存储每个周期的特征
cycle_features = []

# 获取数据中的所有唯一周期
unique_cycles = data['Cycle'].unique()

# 遍历每个周期，提取相应的特征
for cycle in unique_cycles:
    cycle_data = data[data['Cycle'] == cycle]

    # 筛选出恒流（CC）模式的数据 - 电流接近1.5A
    cc_phase = cycle_data[(cycle_data['C_charge'] >= (cc_current_threshold - 0.1)) & (cycle_data['C_charge'] <= (cc_current_threshold + 0.1))]

    # 筛选出恒压（CV）模式的数据 - 电压接近4.2V，并且电流小于20mA
    cv_phase = cycle_data[(cycle_data['V_charge'] >= (cv_voltage_threshold - 0.05)) & (cycle_data['V_charge'] <= (cv_voltage_threshold + 0.05)) &
                           (cycle_data['C_charge'] <= cv_current_threshold)]

    # 计算恒流（CC）时间
    cc_time = cc_phase['T_charge'].iloc[-1] - cc_phase['T_charge'].iloc[0] if not cc_phase.empty else 0

    # 计算恒压（CV）时间
    cv_time = cv_phase['T_charge'].iloc[-1] - cv_phase['T_charge'].iloc[0] if not cv_phase.empty else 0

    # 计算总充电时间
    total_time = cycle_data['T_charge'].iloc[-1] - cycle_data['T_charge'].iloc[0]

    # 计算恒流充电时间占总充电时间的比例
    cc_time_percentage = (cc_time / total_time) * 100 if total_time > 0 else 0

    # 计算恒流（CC）阶段电流-时间曲线包围的面积（积分）
    cc_area = (cc_phase['C_charge'] * (cc_phase['T_charge'].diff())).sum() if not cc_phase.empty else 0

    # 计算恒压（CV）阶段电流-时间曲线包围的面积（积分）
    cv_area = (cv_phase['C_charge'] * (cv_phase['T_charge'].diff())).sum() if not cv_phase.empty else 0

    # 将每个周期的结果存入字典
    cycle_features.append({
        "Cycle": cycle,
        "CC Time (s)": cc_time,
        "CV Time (s)": cv_time,
        "CC Time Percentage (%)": cc_time_percentage,
        "CC Area (A*s)": cc_area,
        "CV Area (A*s)": cv_area
    })

# 将所有周期的特征存入DataFrame
cycle_features_df = pd.DataFrame(cycle_features)

# 保存结果为CSV文件
output_file_all_cycles = 'B18_charging_features.csv'
cycle_features_df.to_csv(output_file_all_cycles, index=False)

