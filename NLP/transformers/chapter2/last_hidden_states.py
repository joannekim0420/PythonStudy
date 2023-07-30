import torch
from transformers import AutoModel
from transformers import DistilBertTokenizer


model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

text = "this is a test"
inputs = tokenizer(text, return_tensors="pt").to(device)
print(inputs)
# {'input_ids': tensor([[ 101, 2023, 2003, 1037, 3231,  102]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]], device='cuda:0')}
print(inputs['input_ids'].size())
# torch.Size([1, 6])


with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
# BaseModelOutput(last_hidden_state=tensor([[[-0.1565, -0.1862,  0.0528,  ..., -0.1188,  0.0662,  0.5470],
#          [-0.3575, -0.6484, -0.0618,  ..., -0.3040,  0.3508,  0.5221],
#          [-0.2772, -0.4459,  0.1818,  ..., -0.0948, -0.0076,  0.9958],
#          [-0.2841, -0.3917,  0.3753,  ..., -0.2151, -0.1173,  1.0526],
#          [ 0.2661, -0.5094, -0.3180,  ..., -0.4203,  0.0144, -0.2149],
#          [ 0.9441,  0.0112, -0.4714,  ...,  0.1439, -0.7288, -0.1619]]],
#        device='cuda:0'), hidden_states=None, attentions=None)

print("outputs.last_hidden_state.size()",outputs.last_hidden_state.size())
#torh.Size([1,6,768])

print(outputs.last_hidden_state[:,0].size())
# torch.Size([1, 768])

def extract_hidden_states(inputs):
    inputs = inputs.to(device)
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}   #CLS token
