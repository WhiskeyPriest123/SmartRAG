import pyarrow.parquet as pq
import csv

def read_parquet(filename):
    parquet_file = pq.ParquetFile(filename)
    
    for batch in parquet_file.iter_batches(batch_size=1):
        df = batch.to_pandas()
        
        for index, row in df.iterrows():
            yield row.to_dict()
        

