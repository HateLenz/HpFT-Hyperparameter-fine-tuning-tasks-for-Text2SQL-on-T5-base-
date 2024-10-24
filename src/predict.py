import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_DIR = '../models/saved_model/checkpoint-3161'

def predict(question):
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model.eval()  # Ensure the model is in evaluation mode

    # Inference
    try:
        input_text = f"translate to SQL: {question}"  # Adjust prompt as per training
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs,
                                 max_length=64,
                                 no_repeat_ngram_size=3,  # 防止重复 n-gram
                                 repetition_penalty=2.0,  # 惩罚重复生成
                                 temperature=0.7,         # 控制生成的随机性
                                 top_p=0.9                # 只从前 90% 概率的词中生成
        )
        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        predicted_sql = "Error generating SQL"
    return predicted_sql

def main():
    question = "What are the names of all the students?"
    sql = predict(question)
    print(f"Question: {question}")
    print(f"Predicted SQL: {sql}")


    questions = [
        "What are the names of the scientists, and how many projects are each of them working on?",
        "What are the names and ids of documents that have the type code BK?",
        "Find the code of city where most of students are living in.",
        "Who is the nominee who has been nominated for the most musicals?",
        "What are the distinct classes that races can have?"
    ]
    correct_answers = [
        "SELECT count(*) ,  T1.name FROM scientists AS T1 JOIN assignedto AS T2 ON T1.ssn  =  T2.scientist GROUP BY T1.name",
        "SELECT document_name ,  document_id FROM Documents WHERE document_type_code  =  \"BK\"",
        "SELECT city_code FROM student GROUP BY city_code ORDER BY count(*) DESC LIMIT 1",
        "SELECT Nominee FROM musical GROUP BY Nominee ORDER BY COUNT(*) DESC LIMIT 1",
        "SELECT DISTINCT CLASS FROM race"
    ]
    for question, answer in zip(questions, correct_answers):
        predicted_sql = predict(question)
        print(f"Question: {question}")
        print(f"Predicted SQL: {predicted_sql}")
        print(f"Correct SQL: {answer}")
        print()
if __name__ == '__main__':
    main()
