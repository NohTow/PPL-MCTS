# Which Discriminator for Cooperative Text Generation?🏇
This is the repository for the code of the  **[Which Discriminator for Cooperative Text Generation?](https://arxiv.org/abs/2204.11586)** paper.

It is an extension of the code of the **[PPL-MCTS: Constrained Textual Generation Through Discriminator-Guided Decoding](https://arxiv.org/pdf/2109.13582.pdf)** paper, which is at the root of this [repository](https://github.com/NohTow/PPL-MCTS/). PPL-MCTS is a plug-and-play decoding method that uses Monte Carlo Tree Search to find a sequence that satisfy a constraint defined by a discriminator. It can be used to guide **any** language model with **any** discriminator that verify if the input sequence satisfy the desired constraint or not.

In this follow-up paper, we studied three types of transformer-based discriminators which offer different capacity/complexity trade-off (by decreasing order of complexity):

 - [Discriminators with bidirectional attention](https://arxiv.org/pdf/1810.04805) 
 - [Discriminators with unidirectional attention](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
 - [Generative discriminators](https://arxiv.org/abs/2009.06367)

Results show that while transformers with bidirectional attention are usually preferred for discriminative tasks, they are not auto-regressive and are therefore much more expensive when used to guide generation. 
Although a little less precise, unidirectional transformers allow to achieve very similar results for a much more reasonable and consistent cost. As a consequence, our study shows that unidirectional discriminators should be preferred for cooperative generation, for which slight accuracy drops can be balanced by reinvesting part of the computational gain.
![image](https://user-images.githubusercontent.com/38869395/156791117-315b0a15-d85f-44f9-b4d9-7c1106aee816.png)


**Since original PPL-MCTS used bidirectional attention, we encourage the use of the unidirectional version of this follow-up, which result in a massive gain in generation time.** 

Generative discriminators, score the whole vocabulary at once. Given the size of usual vocabularies, they seem very interesting at first glance to allow wider search. However, while achieving similar results in terms of classification accuracy, scoring the whole vocabulary comes at the price of a less informative signal. Moreover, although counter-intuitive, this width is not necessarily useful as shown by the search performed by the state-of-the-art Monte Carlo Tree Search, which usually explores more in depth than in width.  Thus, such models will prove useful when used with methods that make particular use of this width information.
We hope the publication of our implementation of GeDi and MCTS version that leverage it will help further experimentations on these promising discriminators for cooperative generation.

## Setup :wrench:
On a Python 3 installation (tested in 3.8), simply install the dependencies defined in the requirement.txt file using

    pip install -r requirements.txt

In order for Generative Discriminators to work, we need to get the language modeling loss for each token. Since by default, Hugging Face LMHeadModels *reduce* the loss (sum/average over every tokens), we need to add `reduction="none"` to the loss function of the LMHeadModel of the model used.
 (e.g, for BERT model, in the class `BertLMHeadModel`, the loss line 1216 `loss_fct =  CrossEntropyLoss()` need to be `loss_fct =  CrossEntropyLoss(reduction="none")`.
 Also, in the same class, please comment the following to allow the use of cached hidden states when using Generative Discriminators:

    if labels is not None:
	    use_cache = False

## Execution parameters
A number of parameters can be defined when using MCTS generation scripts.
|Parameter | Definition |
|--|--|
|\-\-c   |  The exploration constant (c_puct)|
|\-\-alpha   |  The alpha parameter that guide the exploration toward likelihood or value|
|\-\-temperature  |  Language model temperature when calculating priors|
|\-\-penalty   |  Value of the repetition penalty factor defined in the [CTRL paper](https://arxiv.org/abs/1909.05858)|
|\-\-num_it  |  Number of MCTS iteration for one token|
|\-\-batch_size  |  Batch size|

To run usual parameters generation, use: `python mcts_ag_bert_uni.py--temperature 1 --penalty 1.2 --c 3 --num_it 50`

## Training
### Discriminators
A number of parameters can be defined when training discriminators. Training scripts are based based on HF trainer, hence they mostly refer to HF trainer scripts ones.
|Parameter | Definition |
|--|--|
|\-\-model_name_or_path   |  Directory containing the base model or reference to HF model (e.g: bert-based-cased)|
|\-\-train_file   |  Csv file containing a "text" column and corresponding "label" column for training data|
|\-\-validation_file   |  Csv file containing a "text" column and corresponding "label" column for validation data|
|\-\-per_device_train_batch_size  |  Batch size for training|
|\-\-per_device_eval_batch_size  |  Batch size for evaluation|
|\-\-gradient_accumulation_steps   |  Number of gradient accumulations steps to emulate bigger batch size|
|\-\-num_train_epochs |  Number of training epochs|
|\-\-output_dir  |  Directory to save models|

Working example:
 `python classifier_uni_ag.py --model_name_or_path bert-base-cased --train_file datasets/ag_news/full/train_1.csv --validation_file datasets/ag_news/full/validate_1.csv --do_train --do_eval --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --gradient_accumulation_steps 32 --num_train_epochs=20 --output_dir /srv/tempdd/achaffin/bert_bidi_agnews --evaluation_strategy steps --eval_steps 375 --logging_steps 375 --save_steps 375 --ignore_data_skip --preprocessing_num_workers 32`
 
### Language models
The training of the LM is pretty similar to the one of discriminators since they are all based on HuggingFace trainer.
The main difference is that `--train_file` and `--validation_file` should not be a csv with labels but just plain text with one sample by line ending with  `<|endoftext|>` token (and optionally starting with `<|startoftext|>`).

Please note that, in order to use the same tokenizer (rather than decoding the LM and encoding it for the discriminator), the language model is a BERT model. From our experiments, it is way easier to obtain good classification accuracy with pretrained GPT-2 rather than obtaining low perplexity with pre-trained BERT model. 
**Hence, it is surely preferable to use GPT as both generator and discriminator.** 
However, for our experiments, it was easier to make a bidirectional model unidirectional rather than the other way around.

To use GPT-based models rather than BERT ones, use `--model_name_or_path gpt2`. (for discriminators, please change references from BERT to GPT `e.g: BertModel => GPT2Model` and adjust hidden_state size)

## References
Until the pre-print is published, please cite PPL-MCTS if you use this code:
```
@article{DBLP:journals/corr/abs-2204-11586,
  author    = {Antoine Chaffin and Thomas Scialom and Sylvain Lamprier and Jacopo Staiano and Benjamin Piwowarski and Ewa Kijak and Vincent Claveau},
  title     = {{Which Discriminator for Cooperative Text Generation?}},
  journal   = {CoRR},
  volume    = {abs/2204.11586},
  year      = {2022},
  url       = {https://arxiv.org/abs/2204.11586},
  eprinttype = {arXiv},
  eprint    = {2204.11586},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2204-11586.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
