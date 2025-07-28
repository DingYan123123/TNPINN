import pandas as pd

# 加载数据
file_path = '../Mutil NASA/B0005.csv'
data = pd.read_csv(file_path)


# 定义一个函数，用于处理每一列的异常值
def clean_column(column):
    # 描述性统计
    statistics = column.describe()
    q1 = statistics['25%']
    q3 = statistics['75%']
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    median = statistics['50%']

    # 替换异常值为中位数
    return column.apply(lambda x: median if x < lower_bound or x > upper_bound else x)


# 清洗所有数值型列
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col] = clean_column(data[col])

# 保存清洗后的数据为CSV文件
output_file_path = '../NASA Cleaned/B0005_cleaned.csv'
data.to_csv(output_file_path, index=False)
