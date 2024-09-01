from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import  AutoTokenizer
import torch



class DQE(): 
    def __init__(self, model_save_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        self.model = AutoModel.from_pretrained(model_save_path).to(self.device)

    def embed(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embeddings

    def score(self, query, doc):
        with torch.no_grad():
            query_embeddings = self.embed(query)
            doc_embeddings = self.embed([doc])
            query_embeddings = query_embeddings.expand_as(doc_embeddings)
            similarities_list = F.cosine_similarity(query_embeddings, doc_embeddings, dim=1).tolist()
            return  similarities_list[0] if similarities_list.index(max(similarities_list)) == 0 else 0



        

# if __name__ == '__main__':
#     model = USBert(model_path)
#     query = 'Where the capital of Spain'
#     corpus = "Berlin is the capital of Germany."
    
#     score = model.score(query,corpus)
#     print(score)
#     print(type(score))

    
