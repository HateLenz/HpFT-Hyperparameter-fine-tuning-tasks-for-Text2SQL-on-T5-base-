import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

TRAIN_FILE = '../data/raw/train_spider.json'
TABLES_FILE = '../data/raw/tables.json'
OUTPUT_CSV = '../data/processed/train_data.csv'
SPLITS_DIR = '../data/splits/'


# 加载Spider数据集和表结构文件
def load_data(train_file, tables_file):
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    return train_data, tables


# 处理训练数据，将自然语言问题和SQL对加入数据库表结构信息
def preprocess_data(train_data, tables):
    tables_dict = {table['db_id']: table for table in tables}
    processed_data = []

    for item in train_data:
        question = item['question']
        query = item['query']
        db_id = item['db_id']
        db_info = tables_dict.get(db_id, {})
        table_names = [table for table in db_info.get('table_names_original', [])]
        columns = [f"{col[1]} (Table: {table_names[col[0]]})" for col in db_info.get('column_names_original', [])]
        schema_description = " | ".join(columns)

        processed_data.append({
            'question': question,
            'query': query,
            'schema': schema_description
        })

    return pd.DataFrame(processed_data)


# 数据清理函数
def clean_data(df):
    # 删除重复的问题
    df = df.drop_duplicates(subset=['question'], keep='first')

    # 检查空值并删除否则进行补全
    df = df.dropna().reset_index(drop=True)

    return df


# 主函数，执行数据预处理并保存为CSV文件
def main():
    # 加载数据并进行预处理
    train_data, tables = load_data(TRAIN_FILE, TABLES_FILE)
    df = preprocess_data(train_data, tables)

    # 数据清理
    df = clean_data(df)

    # 创建processed目录（如果不存在）
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 保存处理后的数据
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Processed data saved to {OUTPUT_CSV}")

    # 创建splits目录（如果不存在）
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # 划分数据集为训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(SPLITS_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, 'val_split.csv'), index=False)
    print(f"Training and validation data saved to {SPLITS_DIR}")


if __name__ == '__main__':
    main()
