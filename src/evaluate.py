import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sqlparse import format as sql_format
from difflib import SequenceMatcher
import sqlparse
import sqlparse.sql
import sqlparse.tokens as T

MODEL_DIR = '../models/saved_model'
VAL_FILE = '../data/splits/val_split.csv'
LOG_FILE = '../models/evaluation_log.txt'

# 设置日志记录
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_tokens(sql):
    """
    提取 SQL 查询中的重要标记（如 SELECT, FROM, WHERE 等）。
    """
    parsed = sqlparse.parse(sql)
    tokens = []
    for stmt in parsed:
        for token in stmt.flatten():
            if token.ttype in (T.Keyword, T.Name, T.Literal.Number, T.Literal.String.Single, T.Operator):
                tokens.append(token.value.lower())
    return tokens

def sql_match(predicted_sql, actual_sql):
    """
    判断两个 SQL 查询是否逻辑等价。
    使用 sqlparse 格式化 SQL，并比较重要的 SQL 标记。
    """
    # 格式化 SQL
    predicted_sql = sql_format(predicted_sql, reindent=True, keyword_case='lower')
    actual_sql = sql_format(actual_sql, reindent=True, keyword_case='lower')

    # 去除多余的空格和换行符
    predicted_sql = " ".join(predicted_sql.split())
    actual_sql = " ".join(actual_sql.split())

    # 提取重要标记
    predicted_tokens = extract_tokens(predicted_sql)
    actual_tokens = extract_tokens(actual_sql)

    # 比较 SELECT, FROM, WHERE, GROUP BY, ORDER BY 等子句是否相同
    clauses = ['select', 'from', 'where', 'group by', 'order by', 'having']
    for clause in clauses:
        predicted_clause = extract_clause(predicted_sql, clause)
        actual_clause = extract_clause(actual_sql, clause)
        if predicted_clause != actual_clause:
            return False

    # 计算相似度
    similarity = SequenceMatcher(None, predicted_tokens, actual_tokens).ratio()
    return similarity > 0.9  # 调低相似度阈值以提高匹配的严格性

def extract_clause(sql, clause):
    """
    提取 SQL 查询中的特定子句（如 SELECT, FROM, WHERE 等）。
    """
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        tokens = stmt.tokens
        found_clause = False
        clause_tokens = []
        for token in tokens:
            if found_clause:
                if token.is_keyword and token.value.lower() in clause:
                    break
                clause_tokens.append(token)
            if token.is_keyword and token.value.lower() == clause:
                found_clause = True
        if found_clause:
            return " ".join([token.value for token in clause_tokens]).strip()
    return ""

def main():
    # 加载模型和验证数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    val_dataset = load_dataset('csv', data_files=VAL_FILE)['train']
    val_dataset = val_dataset.select(range(50))  # 选择一定数量的模板进行验证

    # 评估模型
    correct = 0
    total = len(val_dataset)
    y_true = []
    y_pred = []

    for sample in val_dataset:
        inputs = tokenizer("translate: " + sample['question'], return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=5,  # 使用 5 个 beams 进行解码，提高生成质量
            early_stopping=True,
            no_repeat_ngram_size=3,  # 阻止重复生成的 n-gram，提高生成质量
            repetition_penalty=2.0,  # 惩罚重复生成
            temperature=0.7,  # 控制生成的随机性
            top_p=0.9
        )

        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        actual_sql = sample['query']

        # 记录详细信息
        logging.info(f"Question: {sample['question']}")
        logging.info(f"Predicted SQL: {predicted_sql}")
        logging.info(f"Actual SQL: {actual_sql}\n")
        print(f"Question: {sample['question']}")
        print(f"Predicted SQL: {predicted_sql}")
        print(f"Actual SQL: {actual_sql}\n")

        # 记录评估数据
        y_true.append(actual_sql.strip().lower())
        y_pred.append(predicted_sql.strip().lower())

        if sql_match(predicted_sql, actual_sql):
            correct += 1

    # 计算精确度
    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    logging.info(f"Validation Accuracy: {accuracy:.2f}%")

    # 计算精确率、取回率、F1分数
    y_true_binary = [1 if sql_match(pred, true) else 0 for pred, true in zip(y_pred, y_true)]
    y_pred_binary = [1 if sql_match(pred, true) else 0 for pred, true in zip(y_pred, y_true)]

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=1) * 100
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=1) * 100
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=1) * 100

    # 打印和记录精确率、取回率和F1分数
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")

    logging.info(f"Precision: {precision:.2f}%")
    logging.info(f"Recall: {recall:.2f}%")
    logging.info(f"F1 Score: {f1:.2f}%")

if __name__ == '__main__':
    main()
