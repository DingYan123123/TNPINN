import pandas as pd

# 载入三个CSV文件
charging_df = pd.read_csv('./B5_charging_features.csv')
discharge_df = pd.read_csv('./B5_discharge_features.csv')
ic_peaks_df = pd.read_csv('./B5_ic_peaks.csv')

# 打印每个DataFrame的前几行进行检查
print("Charging DataFrame:")
print(charging_df.head())

print("\nDischarge DataFrame:")
print(discharge_df.head())

print("\nIC Peaks DataFrame:")
print(ic_peaks_df.head())

# 合并三个DataFrame，按'Cycle'列合并
merged_df = pd.merge(charging_df, ic_peaks_df, on='Cycle', how='outer')
merged_df = pd.merge(merged_df, discharge_df, on='Cycle', how='outer')
merged_df.drop(columns=['disCV Time (s)'], inplace=True)

# 显示合并后的DataFrame前几行
print("\nMerged DataFrame:")
print(merged_df.head())

# 保存合并后的数据为CSV文件
output_file = 'B5.csv'
merged_df.to_csv(output_file, index=False)

print(f"数据已成功保存到 {output_file}")
