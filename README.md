# PPL-MCTS
This is the repository for the code of the **[PPL-MCTS: Constrained Textual Generation Through Discriminator-Guided Decoding](https://arxiv.org/pdf/2109.13582.pdf)** paper.

It is a plug-and-play decoding method that uses Monte Carlo Tree Search to find a sequence that satisfy a constraint defined by a discriminator. It can be used to guide **any** language model with **any** discriminator that verify if the input sequence satisfy the desired constraint or not.
The code is based on the [Hugging Face transformers library](https://huggingface.co/docs/transformers/index) both for the language model and the discriminator.
## Setup :wrench:
On a Python 3 installation (tested in 3.8), simply install the dependencies defined in the requirement.txt file using

    pip install -r requirements.txt
## Execution parameters
A number of parameters can be defined when executing the MCTS.
|Parameter | Definition |
|--|--|
|\-\-c   |  The exploration constant (c_puct)|
|\-\-alpha   |  The alpha parameter that guide the exploration toward likelihood or value|
|\-\-temperature  |  Language model temperature when calculating priors|
|\-\-penalty   |  Value of the repetition penalty factor defined in the [CTRL paper](https://arxiv.org/abs/1909.05858)|
|\-\-num_it  |  Number of MCTS iteration for one token|
|\-\-rollout_size  |  Number of tokens to generate during rollout|
|\-\-batch_size  |  Batch size|

## Code walkthrough
Some lines of code can be adjusted for your usage.
Define the path were Hugging Face models and data will be stored
```python
os.environ['TRANSFORMERS_CACHE'] =
```
Import desired architecture for language model and discriminator and corresponding tokenizers.
```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RepetitionPenaltyLogitsProcessor, FlaubertTokenizer, FlaubertModel
```
(Optional, you can replace the discriminator by a HF vanilla one as the language model below)

Define the downstream architecture of the discriminator an import discriminator weights
```python
#-------------- Model definition ---------------#

class  Net(nn.Module):
	
	def  __init__(self):
		super(Net,  self).__init__()
		self.flaubert = FlaubertModel.from_pretrained('flaubert/flaubert_large_cased',  		output_hidden_states=True)
		self.fc_classif = nn.Linear(1024,  2)

	def  forward(self, texts):
		tokenizer_res = tokenizer.batch_encode_plus(texts,  truncation=True,  max_length=512,  padding='longest')
		tokens_tensor = torch.cuda.LongTensor(tokenizer_res['input_ids'])
		attention_tensor = torch.cuda.LongTensor(tokenizer_res['attention_mask'])
		output =  self.flaubert(tokens_tensor,  attention_mask=attention_tensor)
		text = F.normalize(torch.div(torch.sum(output[1][-1],  axis=1),torch.unsqueeze(torch.sum(attention_tensor,  axis=1),1)))
		text =  self.fc_classif(text)
		return nn.Softmax(dim  =  1)(text).cpu()

model_path =  "../datasets/flue/CLS/models/validation_BEST_bert_tuned_2021_08_19-17_58_19.pth"
```
Import language model weights
```python
gpt = GPT2LMHeadModel.from_pretrained("../../gpt2-cls/best")
tokenizer_gpt = GPT2TokenizerFast.from_pretrained("../../gpt2-cls/best")
```
**Define the function that take sequences token ids and returns probability the belong to the target class for each sequence**
```python
def  get_values(tokens_ids, labels):
	"""Gets sequence scores from the discriminator"""
	propositions = tokenizer_gpt.batch_decode(tokens_ids,  skip_special_tokens=True,  clean_up_tokenization_spaces=True)
	with torch.no_grad():
		outputs =  net(propositions)
		return outputs[labels]
```
Define prompts to fill using MCTS and corresponding labels
```python
for i, (_, row) in  enumerate(lines.iterrows()):
	labels[i, int(row["label"])] =  1
	prompt_texts[i] =  "<|startoftext|> "  +  str(row["text"])
```



