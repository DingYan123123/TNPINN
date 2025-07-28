import pandas as pd

# 加载放电数据
discharge_file_path = '../NASA/B0005discharge.csv'
discharge_data = pd.read_csv(discharge_file_path)

# 初始化一个空列表来存储每个周期的特征
discharge_cycle_features = []

# 设置恒流和恒压模式的阈值
cc_current_threshold = -2.0  # A (恒流放电模式电流阈值, 负值表示放电)
cv_voltage_threshold = 2.7  # V (恒压模式电压阈值)

# 获取放电数据中的所有唯一周期
unique_cycles_discharge = discharge_data['Cycle'].unique()

# 遍历每个周期，提取相应的特征
for cycle in unique_cycles_discharge:
    cycle_data = discharge_data[discharge_data['Cycle'] == cycle]

    # 筛选出恒流（CC）放电阶段的数据 - 电流接近-2A
    cc_phase = cycle_data[(cycle_data['C_charge'] >= (cc_current_threshold - 0.1)) &
                           (cycle_data['C_charge'] <= (cc_current_threshold + 0.1))]

    # 筛选出恒压（CV）放电阶段的数据 - 电压下降至2.5V或以下
    cv_phase = cycle_data[cycle_data['V_charge'] <= cv_voltage_threshold]

    # 计算恒流放电时间
    cc_time = cc_phase['T_charge'].iloc[-1] - cc_phase['T_charge'].iloc[0] if not cc_phase.empty else 0

    # 计算恒压放电时间
    cv_time = cv_phase['T_charge'].iloc[-1] - cv_phase['T_charge'].iloc[0] if not cv_phase.empty else 0

    # 获取电池在当前周期的电量
    capacity = cycle_data['Capacity'].iloc[0]  # 以第一个数据点的容量为该周期的电量

    # 将当前周期的特征存入字典
    discharge_cycle_features.append({
        "Cycle": cycle,
        "disCC Time (s)": cc_time,
        "disCV Time (s)": cv_time,
        "Capacity (Ah)": capacity
    })

# 将提取的特征保存到DataFrame中
discharge_cycle_features_df = pd.DataFrame(discharge_cycle_features)

# 保存结果为CSV文件
output_file_discharge_with_capacity = 'B5_discharge_features.csv'
discharge_cycle_features_df.to_csv(output_file_discharge_with_capacity, index=False)

