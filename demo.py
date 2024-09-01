from Utils.read_file import read_parquet




if __name__ == '__main__':
    file = "Dataset/DROP/train-00000-of-00001.parquet"
    file_data = read_parquet(file)
    count = 0
    for d in file_data:
        for i in d.keys():
            print(i,'-'*10,d[i])
        count += 1
        print("*"*50)
        
        if count > 2:
            break