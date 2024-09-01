
from Utils.read_file import read_parquet, append_to_tsv, append_to_json, write_list_to_json_line_by_line
from tqdm import tqdm
import random



if __name__ == '__main__':
    train_file = "Dataset/DROP/train-00000-of-00001.parquet"
    val_file = 'Dataset/DROP/validation-00000-of-00001.parquet'
    
    document_file = 'Dataset/DROP/document.tsv'
    query_file = 'Dataset/DROP/query.tsv'
    answer_file = 'Dataset/DROP/answer.tsv'
    
    train_rel_file = 'Dataset/DROP/train.json'
    val_rel_file = 'Dataset/DROP/val.json'
    
    train_data = read_parquet(train_file)
    val_data = read_parquet(val_file)
    document_dict = {}
    query_dict = {}
    d2answer = {}
    train_q2d = []
    val_q2d = []
    

    for data in tqdm(train_data):
        d_id = data['section_id'] 
        q_id = data['query_id']
        document = data['passage']
        query = data['question']
        answer = data['answers_spans']['spans']
        document_dict[d_id] = document
        query_dict[q_id] = query
        d2answer[d_id] = answer
        
        train_q2d.append([q_id, d_id])
    
    for data in tqdm(val_data):
        d_id = data['section_id'] 
        q_id = data['query_id']
        document = data['passage']
        query = data['question']
        answer = data['answers_spans']['spans']
        
        
        document_dict[d_id] = document
        query_dict[q_id] = query
        d2answer[d_id] = answer
        
        val_q2d.append([q_id, d_id])
    
    
    
    for id, value in enumerate(train_q2d):
        qid, did = value
        selected = random.choice(list(document_dict.keys()))
        while selected == value:
            selected = random.choice(list(document_dict.keys()))
        train_q2d[id] = [qid, did, selected]
    
    for id, value in enumerate(val_q2d):
        qid, did = value
        selected = random.choice(list(document_dict.keys()))
        while selected == value:
            selected = random.choice(list(document_dict.keys()))
        val_q2d[id] = [qid, did, selected]
    
    
    
    
    

    append_to_tsv(document_file,document_dict)
    append_to_tsv(query_file,query_dict)
    append_to_tsv(answer_file,d2answer)

    write_list_to_json_line_by_line(train_rel_file, train_q2d)
    write_list_to_json_line_by_line(val_rel_file, val_q2d)

    
    
