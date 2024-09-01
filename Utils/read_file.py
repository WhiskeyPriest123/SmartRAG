import pyarrow.parquet as pq
import csv
import json

def read_parquet(filename):
    parquet_file = pq.ParquetFile(filename)
    for batch in parquet_file.iter_batches(batch_size=1):
        df = batch.to_pandas()
        for index, row in df.iterrows():
            yield row.to_dict()
        

def append_to_tsv(file_path, idstext):
    with open(file_path, 'a', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for id1 in idstext.keys():
            writer.writerow([id1, idstext[id1]])


def append_to_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
