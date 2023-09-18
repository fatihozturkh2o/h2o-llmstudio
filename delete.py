from transformers import AutoModel, AutoTokenizer
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


print("hello!")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
model.to('cuda:0')
# # Define a list of text samples you want to classify
# text_samples = [
#     "Merhaba naber",
# ]

# # Tokenize the input text and convert it to a tensor
# input_ids = tokenizer(text_samples, padding=True, truncation=True, return_tensors="pt")
# input_ids.to('cuda:0')  # Send the input to GPU if available
# print(input_ids)

# # Perform inference (get predictions)
# print("model output....")
# with torch.no_grad():
#     outputs = model(**input_ids)
# print(outputs[0].shape)

# sentence_embeddings = mean_pooling(outputs, input_ids['attention_mask'])
# print("Sentence embeddings:")
# print(sentence_embeddings.shape)


from transformers import pipeline
unmasker = pipeline('fill-mask', model='dbmdz/bert-base-turkish-cased')
print(unmasker("Dünyanın en iyi filmi: [MASK] filmidir."))

