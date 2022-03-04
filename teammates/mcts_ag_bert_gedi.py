import os
os.environ['TRANSFORMERS_CACHE'] = '/srv/tempdd/achaffin/.cache'
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertLMHeadModel, BertTokenizerFast, BertConfig, RepetitionPenaltyLogitsProcessor
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import argparse
import logging
import random
parser = argparse.ArgumentParser()
parser.add_argument(
    "--c",
    default=None,
    type=float,
    required=True,
    help="The exploration constant"
)
parser.add_argument(
    "--alpha",
    default=1,
    type=float,
    help="The parameter that guide the exploration toward likelihood or value"
)
parser.add_argument(
    "--temperature",
    default=None,
    type=float,
    required=True,
    help="Temperature when calculating priors"
)

parser.add_argument(
    "--penalty",
    default=1.0,
    type=float,
    help="Penalty factor to apply to repetitions"
)

parser.add_argument(
    "--num_it",
    default=50,
    type=int,
    required=False,
    help="Number of MCTS iteration for one token"
)


args = parser.parse_args()


logging.basicConfig(
    format="%(message)s",
    level=logging.WARNING,
    filename=("log/ag_news/mcts_{}_{}_{}_{}_gedi.log".format(args.c, args.temperature, args.penalty, args.num_it))
)
logger = logging.getLogger(__name__)


print("loading LM")
config = BertConfig.from_pretrained("/srv/tempdd/achaffin/bert_lm_ag")
config.is_decoder = True
config.use_cache = True
lm = BertLMHeadModel.from_pretrained("bert_lm_ag", config=config)
lm.eval()
lm.to("cuda")
tokenizer = BertTokenizerFast.from_pretrained("bert_lm_ag")
tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token 
# eos_token_id = lm.config.eos_token_id
pad_token_id = lm.config.pad_token_id
vocab_size = tokenizer.vocab_size
print("LM loaded")

print("loading GeDi model")
gedi_conf = BertConfig.from_pretrained("bert_classloss_agnews_relembkey")
gedi_conf.is_decoder = True
gedi_conf.use_cache = True
gedi_conf.position_embedding_type = "relative_key"
gedi = BertLMHeadModel.from_pretrained("bert_classloss_agnews_relembkey", config = gedi_conf)
gedi.eval()
gedi.to("cuda")
print("GeDi model loaded")

# The maximum length the LM and/or the classifier can handle. 512 in case of BERT
MAX_SEQUENCE_LENGTH = 512
label_names = ["Wor", "Spo", "Bus", "Sci"]
LABEL_LENGTH = len(tokenizer("[" + label_names[0] + "]", return_tensors="pt", add_special_tokens=False))

from typing import Optional
if not os.path.exists("log"):
    os.makedirs("log")

cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
def pad_sequences_to_left(sequences, batch_first=False, padding_value=0):
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, max_len-length:, ...] = tensor
        else:
            out_tensor[max_len-length:, i, ...] = tensor
    return out_tensor


def pad_sequences_to_left_states(sequences, padding_value=0, max_len=0):
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (max_size[0], max_size[1], len(sequences), max_size[2], max_len, max_size[4])
    out_tensor = sequences[0].new_full(out_dims, padding_value, device="cuda")
    for i, tensor in enumerate(sequences):
        length = tensor.size()[3]
        out_tensor[:, :, i, :, max_len-length:, ...] = tensor
    return out_tensor


def root_fun(original_input, cons_input_ids, cons_attention_masks, labels, temperature, repetition_penalty):
    # Forward pass of LM to get priors and states
    model_inputs = lm.prepare_inputs_for_generation(original_input.input_ids, attention_mask=original_input.attention_mask, use_cache=True)
    with torch.no_grad():
        outputs = lm(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )
        states = outputs.past_key_values

        prompt_masked_input_ids = torch.clone(model_inputs["input_ids"])
        inverted_attention_mask = model_inputs["attention_mask"] == 0
        prompt_masked_input_ids[inverted_attention_mask]=14827
        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        priors = F.softmax(priors, dim=-1).cpu().numpy()
        
    
    seq_ids = torch.cat(tuple(cons_input_ids[label_name] for label_name in label_names), dim=0)
    label_tokens = seq_ids.clone()
    seq_masks = torch.cat(tuple(cons_attention_masks[label_name] for label_name in label_names), dim=0)
    gedi_inputs = gedi.prepare_inputs_for_generation(seq_ids, attention_mask=seq_masks)
    gedi_inputs["labels"] = label_tokens
    # Use of our discriminator to get values
    with torch.no_grad():
        gedi_output = gedi(**gedi_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,use_cache=True)
    gen_losses = torch.sum((gedi_output["loss"].view(len(label_names)*original_input["input_ids"].shape[0],-1) * seq_masks[:,1:]) / torch.sum(seq_masks[:,1:],1,keepdim=True), dim=1).view(len(label_names),-1).permute(1, 0)
    child_prob = F.log_softmax(gedi_output["logits"][:,-1,:]).view(len(label_names),original_input["input_ids"].shape[0], -1).permute(1, 0, 2) - gen_losses.view(original_input["input_ids"].shape[0], len(label_names), 1).expand(-1, -1, tokenizer.vocab_size)
    cons_states = gedi_output.past_key_values
    return priors, child_prob, states, cons_states

def rec_fun(original_states, cons_states, original_token_ids, original_attention_masks, cons_input_ids, cons_attention_masks, values, labels, temperature, repetition_penalty):
    # Forward pass of LM to get priors and states
    model_inputs = lm.prepare_inputs_for_generation(original_token_ids, attention_mask=original_attention_masks,  past=original_states)
    with torch.no_grad():
        outputs = lm(
            **model_inputs,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        next_states = outputs.past_key_values

        prompt_masked_input_ids = torch.clone(original_token_ids)
        inverted_attention_mask = original_attention_masks == 0
        #penalizing an unused token
        prompt_masked_input_ids[inverted_attention_mask]=28988
        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)

        priors = F.softmax(priors, dim=-1).cpu().numpy()
    
    seq_ids = torch.cat(tuple(cons_input_ids[label_name] for label_name in label_names), dim=0)
    label_tokens = seq_ids.clone()
    seq_masks = torch.cat(tuple(cons_attention_masks[label_name] for label_name in label_names), dim=0)
    gedi_inputs = gedi.prepare_inputs_for_generation(seq_ids, attention_mask=seq_masks, past=cons_states)
    # Use of our discriminator to get values
    with torch.no_grad():
        gedi_output = gedi(**gedi_inputs, return_dict=True, output_attentions=False, output_hidden_states=False, use_cache=True)
    child_prob = F.log_softmax(gedi_output["logits"][:,-1,:]).view(len(label_names), original_token_ids.shape[0], -1).permute(1, 0, 2) + torch.unsqueeze(values, 2).expand(-1, -1, tokenizer.vocab_size)
    cons_states = gedi_output.past_key_values
    return priors, child_prob, next_states, cons_states



class NumpyMCTS():
    def __init__(self, root_fun, rec_fun, batch_size, num_simulations, num_actions, num_sparse_actions, pb_c_init, temperature, alpha, penalty):
        self._batch_size = batch_size
        self._num_simulations = num_simulations
        self._num_actions = num_actions
        self._num_sparse_actions = min(num_sparse_actions, num_actions)
        self._pb_c_init = pb_c_init
        self._temperature = temperature
        self.alpha = alpha

        self._root_fun = root_fun # a function called at the root
        self._rec_fun = rec_fun # a function called in the tree
        self._adaptive_min_values = np.zeros(batch_size, dtype=np.float32)
        self._adaptive_max_values = np.zeros(batch_size, dtype=np.float32)
        self._labels = torch.zeros((batch_size, len(label_names)), dtype=torch.bool, device="cuda")

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        num_nodes = num_simulations + 1
        batch_node = (batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._likelihoods = np.zeros(batch_node, dtype=np.float32)
        self._raw_values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=np.bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        batch_node_action = (batch_size, num_nodes, self._num_sparse_actions) # (B, )
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_probas = np.zeros((batch_size, num_nodes, self._num_sparse_actions, len(label_names)), dtype=np.float32)
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._states = {}
        self._token_ids = {}
        self._attention_mask = {}
        for label_name in label_names:
            self._states[label_name] = {}
            self._token_ids[label_name] = {}
            self._attention_mask[label_name] = {}
        self._states["original"] = {}
        self._token_ids["original"] = {}
        self._attention_mask["original"] = {}
        self._batch_range = np.arange(batch_size)
        self._reset_tree()
        self._repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty)

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._likelihoods.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_probas.fill(0.0)
        self._children_visits.fill(0)
        self._states = {}
        self._token_ids = {}
        self._attention_mask = {}
        for label_name in label_names:
            self._states[label_name] = {}
            self._token_ids[label_name] = {}
            self._attention_mask[label_name] = {}
        self._states["original"] = {}
        self._token_ids["original"] = {}
        self._attention_mask["original"] = {}
    
    def set_labels(self, labels): 
        self._labels = labels
    def search(self, original_input, cons_input_ids, cons_attention_masks):
        self._reset_tree()

        # Evaluate the root.
        prior, child_prob, states, cons_states = self._root_fun(original_input, cons_input_ids, cons_attention_masks, self._labels, self._temperature, self._repetition_penalty)

       
        self._adaptive_min_values = 1
        self._adaptive_max_values = 1 + 1e-6

        root_index = 0
        self.create_node(root_index, prior, 1, 1, child_prob, states, original_input.input_ids, original_input.attention_mask, cons_states, cons_input_ids, cons_attention_masks, np.full(self._batch_size, False, dtype=np.bool))
        
        # Do simulations, expansions, and backwards.
        leaf_indices = np.zeros((self._batch_size), np.int32)
        existing_nodes = 0
        tokens_to_generate = 98
        tokens_pbar = tqdm(total = tokens_to_generate, desc="Tokens generated")
        for i in range(tokens_to_generate):
            for sim in range(self._num_simulations):
                node_indices, actions = self.simulate()
                next_node_index = sim + 1 + existing_nodes # root is 0, therefore we offset by 1.
                self.expand(node_indices, actions, next_node_index)
                leaf_indices.fill(next_node_index)
                self.backward(leaf_indices)
            visit_counts, _ = self.dense_visit_counts()
            existing_nodes = np.amax(visit_counts)
            # Create new tree with selected node as root
            num_nodes = self._num_simulations + existing_nodes + 1
            batch_node = (self._batch_size, num_nodes)
            temp_visit_counts = np.zeros(batch_node, dtype=np.int32)
            temp_values = np.zeros(batch_node, dtype=np.float32)
            temp_likelihoods = np.zeros(batch_node, dtype=np.float32)
            temp_raw_values = np.zeros(batch_node, dtype=np.float32)
            temp_parents = np.full(batch_node, -1, dtype=np.int32)
            temp_action_from_parents = np.full(batch_node, -1, dtype=np.int32)
            temp_depth = np.zeros(batch_node, dtype=np.int32)
            temp_is_terminal = np.full(batch_node, False, dtype=np.bool)
            batch_node_action = (self._batch_size, num_nodes, self._num_sparse_actions) # (B, )
            temp_topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
            temp_children_index = np.full(batch_node_action, -1, dtype=np.int32)
            temp_children_prior = np.zeros(batch_node_action, dtype=np.float32)
            temp_children_probas = np.zeros((self._batch_size, num_nodes, self._num_sparse_actions, len(label_names)), dtype=np.float32)
            temp_children_values = np.zeros(batch_node_action, dtype=np.float32)
            temp_children_visits = np.zeros(batch_node_action, dtype=np.int32)
            temp_states = {}
            temp_token_ids = {} # Indexed by tuples (batch index, node index)
            temp_attention_mask = {}
            for label_name in label_names:
                temp_states[label_name] = {}
                temp_token_ids[label_name] = {}
                temp_attention_mask[label_name] = {}
            temp_states["original"] = {}
            temp_token_ids["original"] = {}
            temp_attention_mask["original"] = {}

            for b, new_root_action in enumerate(np.argmax(visit_counts,axis=1)):
                new_root_id = self._children_index[b, 0, new_root_action]
                new_node_id = 1
                old_to_new_id = {new_root_id:0}
                children_to_explore = self._children_index[b, new_root_id][self._children_index[b, new_root_id] != -1].tolist()
                while(len(children_to_explore)>0):
                    child_id = children_to_explore.pop(0)
                    old_to_new_id[child_id] = new_node_id
                    children_to_explore += self._children_index[b, child_id][self._children_index[b, child_id] != -1].tolist()
                    new_node_id += 1
                for old_id, new_id in old_to_new_id.items():
                    if(new_id !=0):
                        temp_parents[b, new_id] = old_to_new_id[self._parents[b, old_id]]
                        temp_action_from_parents[b, new_id] = self._action_from_parents[b, old_id]
                    for i, children in enumerate(self._children_index[b, old_id]):
                        if(children != -1):
                            temp_children_index[b, new_id, i] = old_to_new_id[children]
                    temp_visit_counts[b, new_id] = self._visit_counts[b, old_id]
                    temp_values[b, new_id] = self._values[b, old_id]
                    temp_likelihoods[b, new_id] = self._likelihoods[b, old_id]
                    temp_raw_values[b, new_id] = self._raw_values[b, old_id]
                    
                    temp_action_from_parents[b, new_id] = self._action_from_parents[b, old_id]
                    temp_depth[b, new_id] = self._depth[b, old_id] - 1
                    temp_is_terminal[b, new_id] = self._is_terminal[b, old_id]
                
                    temp_topk_mapping[b, new_id] = self._topk_mapping[b, old_id]
                    temp_children_prior[b, new_id] = self._children_prior[b, old_id]
                    temp_children_probas[b, new_id] = self._children_probas[b, old_id]
                    temp_children_values[b, new_id] = self._children_values[b, old_id]
                    temp_children_visits[b, new_id] = self._children_visits[b, old_id]

                    for label_name in label_names:
                        temp_states[label_name][(b, new_id)] = self._states[label_name][(b, old_id)]
                        temp_token_ids[label_name][(b, new_id)] = self._token_ids[label_name][(b, old_id)]
                        temp_attention_mask[label_name][(b, new_id)] = self._attention_mask[label_name][(b, old_id)]
                    temp_states["original"][(b, new_id)] = self._states["original"][(b, old_id)]
                    temp_token_ids["original"][(b, new_id)] = self._token_ids["original"][(b, old_id)]
                    temp_attention_mask["original"][(b, new_id)] = self._attention_mask["original"][(b, old_id)]
                for label_name in label_names:
                    temp_states[label_name][(b, 0)] = torch.cat((self._states[label_name][(b, 0)], self._states[label_name][(b, new_root_id)]), 3)
                temp_states["original"][(b, 0)] = torch.cat((self._states["original"][(b, 0)], self._states["original"][(b, new_root_id)]), 3)
 
            self._num_nodes = num_nodes
            self._visit_counts = temp_visit_counts 
            self._values = temp_values
            self._likelihoods = temp_likelihoods
            self._raw_values = temp_raw_values 
            self._parents = temp_parents
            self._action_from_parents = temp_action_from_parents 
            # The 0-indexed depth of the node. The root is the only 0-depth node.
            # The depth of node i, is the depth of its parent + 1.
            self._depth = temp_depth 
            self._is_terminal = temp_is_terminal
            self._topk_mapping = temp_topk_mapping 
            self._children_index = temp_children_index
            self._children_prior = temp_children_prior 
            self._children_probas = temp_children_probas
            self._children_values = temp_children_values
            self._children_visits = temp_children_visits
            self._states = temp_states
            self._token_ids = temp_token_ids
            self._attention_mask = temp_attention_mask
            tokens_pbar.update(1)
            # If every sequences is finished, stop
            if(self._is_terminal[:, 0].all()):
                break
        for b in range(self._batch_size):
            logging.warning((tokenizer.decode(self._token_ids["original"][(b,0)], skip_special_tokens=False, clean_up_tokenization_spaces=True)).replace("\n","").replace(" [PAD]", "") + "[PAD]")

    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_visit_counts
        return root_visit_counts, dense_visit_counts
    
    def dense_scores(self):
        root_index = 0
        root_scores = self._children_values[:, root_index, :]
        dense_root_scores = np.zeros((self._batch_size, self._num_actions))
        dense_root_scores[self._batch_range[:, None], self._child_prob_mapping[:, root_index, :]] = root_scores
        root_visit_counts = self._children_visits[:, root_index, :]
        return dense_root_scores

    def dense_mean_scores(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        root_scores = self._children_values[:, root_index, :]
        root_mean_scores = root_scores / root_visit_counts
        dense_mean_scores = np.zeros((self._batch_size, self._num_actions))
        dense_mean_scores[self._batch_range[:, None], self._child_prob_mapping[:, root_index, :]] = root_mean_scores
        return dense_mean_scores

    def simulate(self):
        """Goes down until all elements have reached unexplored actions."""
        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        while True:
            depth += 1
            actions = self.uct_select_action(node_indices)
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                return node_indices, actions
            else:
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)
    
    def uct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_prior = self._children_prior[self._batch_range, node_indices, :] # (B, A)
        node_children_values = self._children_values[self._batch_range, node_indices, :] # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :] # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices] # (B)
        node_policy_score = np.sqrt(node_visits[:, None]) * self._pb_c_init * node_children_prior / (node_children_visits + 1) 
        # (B, A)
        
        node_value_score = node_children_values 
        
        node_uct_score = node_value_score + node_policy_score # (B, A)
        actions = np.argmax(node_uct_score, axis=1)
        return actions
    
    # return state
    def get_states_from_node(self, b, n, d, label): 
        state_array = [None] * d
        state_array[d-1] = self._states[label][(b, n)]
        while n!=0:
            n = self._parents[(b, n)]
            d -= 1
            state_array[d-1] = self._states[label][(b, n)]
        return torch.cat(state_array, 3)

    
    

    def expand(self, node_indices, actions, next_node_index):
        """Creates and evaluate child nodes from given nodes and unexplored actions."""
        # Retrieve token ids and masks for nodes to be evaluated.
        original_tokens_ids = pad_sequences_to_left([self._token_ids["original"][(b, n)] for b, n in enumerate(node_indices)], True, pad_token_id)
        original_attention_masks = pad_sequences_to_left([self._attention_mask["original"][(b, n)] for b, n in enumerate(node_indices)], True, 0)
        cons_tokens_ids = {}
        cons_attention_masks = {}
        for label_name in label_names:
            cons_tokens_ids[label_name] = pad_sequences_to_left([self._token_ids[label_name][(b, n)] for b, n in enumerate(node_indices)], True, pad_token_id)
            cons_attention_masks[label_name] = pad_sequences_to_left([self._attention_mask[label_name][(b, n)] for b, n in enumerate(node_indices)], True, 0)
        depths = torch.tensor([self._depth[(b, n)]+1 for b, n in enumerate(node_indices)], device="cuda")
        children_priors = np.array([self._children_prior[(b, n)][actions[b]] for b, n in enumerate(node_indices)])
        likelihoods = np.array([self._likelihoods[(b, n)] for b, n in enumerate(node_indices)])
        values = torch.tensor([self._children_probas[(b, n)][actions[b]] for b, n in enumerate(node_indices)], device="cuda")
        previous_node_is_terminal = self._is_terminal[self._batch_range, node_indices[self._batch_range]] # (B)

        
        original_states_tensor = pad_sequences_to_left_states([self.get_states_from_node(b, n.item(), depths[b].item(), "original") for b, n in enumerate(node_indices)], 0, max_len=len(original_tokens_ids[0]))
        cons_states_tensor = {}
        for label_name in label_names:
            cons_states_tensor[label_name] = pad_sequences_to_left_states([self.get_states_from_node(b, n, depths[b].item(), label_name) for b, n in enumerate(node_indices)], 0, max_len=len(cons_tokens_ids[label_name][0]))
        
        if(len(cons_tokens_ids[label_names[0]][0])>=MAX_SEQUENCE_LENGTH):
            previous_node_is_terminal[torch.sum(cons_attention_masks[label_names[0]], axis=1).cpu()>=MAX_SEQUENCE_LENGTH] = True
            original_tokens_ids = original_tokens_ids[:, -(MAX_SEQUENCE_LENGTH-LABEL_LENGTH-1):]
            original_attention_masks = original_attention_masks[:, -(MAX_SEQUENCE_LENGTH-LABEL_LENGTH-1):]
            original_states_tensor = original_states_tensor[:, :, :, :, -(MAX_SEQUENCE_LENGTH-LABEL_LENGTH-1):]
            for label_name in label_names:
                cons_states_tensor[label_name] = cons_states_tensor[label_name][:, :, :, :, -(MAX_SEQUENCE_LENGTH-1):]
                cons_tokens_ids[label_name] = cons_tokens_ids[label_name][:, -(MAX_SEQUENCE_LENGTH-1):]
                cons_attention_masks[label_name]= cons_attention_masks[label_name][:, -(MAX_SEQUENCE_LENGTH-1):]
        original_states = tuple(tuple(type_of_value for type_of_value in layer) for layer in original_states_tensor)
        cons_states = tuple(tuple(type_of_value for type_of_value in layer) for layer in torch.cat(tuple(cons_states_tensor[label_name] for label_name in label_names), dim=2))
        
        
        
        # Convert sparse actions to dense actions for network computation
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        dense_actions[previous_node_is_terminal] = pad_token_id
        # Add actions to list of tokens and extend attention mask by 1
        original_tokens_ids = torch.cat((original_tokens_ids, torch.unsqueeze(torch.cuda.LongTensor(dense_actions), 1)), dim = 1)
        original_attention_masks = torch.cat((original_attention_masks, torch.unsqueeze(torch.ones(len(dense_actions), dtype=torch.long, device="cuda"), 1)), dim = 1)
        for label_name in label_names:
            cons_tokens_ids[label_name] = torch.cat((cons_tokens_ids[label_name], torch.unsqueeze(torch.cuda.LongTensor(dense_actions), 1)), dim = 1)
            cons_attention_masks[label_name] = torch.cat((cons_attention_masks[label_name], torch.unsqueeze(torch.ones(len(dense_actions), dtype=torch.long, device="cuda"), 1)), dim = 1)
        
        # Check if expanded nodes are terminal 
        expanded_node_is_terminal = np.logical_or((dense_actions == pad_token_id), previous_node_is_terminal)
        # Evaluate nodes.
        (prior, child_prob, next_states, cons_states) = self._rec_fun(original_states, cons_states, original_tokens_ids, original_attention_masks, cons_tokens_ids, cons_attention_masks, values, self._labels, self._temperature, self._repetition_penalty)

        child_prob[previous_node_is_terminal] = torch.unsqueeze(values[previous_node_is_terminal], 2).expand(-1, -1, tokenizer.vocab_size)

        for label_name in label_names:
            cons_attention_masks[label_name] = [torch.cat((self._attention_mask[label_name][(b, n)], torch.ones(1, dtype=torch.long, device="cuda")), dim=0) for b, n in enumerate(node_indices)]
            cons_tokens_ids[label_name] = [torch.cat((self._token_ids[label_name][(b, n)], torch.cuda.LongTensor([dense_actions[b]])), dim=0) for b, n in enumerate(node_indices)]
        original_attention_masks = [torch.cat((self._attention_mask["original"][(b, n)], torch.ones(1, dtype=torch.long, device="cuda")), dim=0) for b, n in enumerate(node_indices)]
        original_tokens_ids = [torch.cat((self._token_ids["original"][(b, n)], torch.cuda.LongTensor([dense_actions[b]])), dim=0) for b, n in enumerate(node_indices)]
        


        # Create the new nodes.
        values = torch.exp(-cross_entropy(values, self._labels)).cpu()
        self.create_node(next_node_index, prior, likelihoods*children_priors, values, child_prob, next_states, original_tokens_ids, original_attention_masks, cons_states, cons_tokens_ids, cons_attention_masks, expanded_node_is_terminal)
        
        # Update the min and max values arrays
        self._adaptive_min_values = np.minimum(self._adaptive_min_values, values)
        self._adaptive_max_values = np.maximum(self._adaptive_max_values, values)
        
        # Update tree topology.
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1

    def create_node(self, node_index, prior, likelihoods, values, child_prob, original_states, original_tokens_ids, original_attention_masks, cons_states, cons_input_ids, cons_attention_mask, expanded_node_is_terminal):
        # Truncate the prior to only keep the top k logits
        prior_topk_indices = np.argpartition(prior, -self._num_sparse_actions, axis=-1)[:, -self._num_sparse_actions:]
        child_prob = child_prob[self._batch_range[:, None], :, prior_topk_indices] # (B, A)
        prior = prior[self._batch_range[:, None], prior_topk_indices] # (B, A)
        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices

        # Update prior, values and visit counts.
        self._children_prior[:, node_index, :] = prior
        self._children_probas[:, node_index, :] = child_prob.cpu()
        self._likelihoods[:, node_index] = likelihoods
        # raw_values = values**(self.alpha) * likelihoods**(1-self.alpha)
        raw_values = values
        self._values[:, node_index] = raw_values
        self._raw_values[:, node_index] = raw_values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal
        # States has shape [12 (nhead), 2(key/value), batch_size, 12(nlayer), seq_len, 64]
        original_key_value_tensor = torch.stack(list(torch.stack(list(original_states[i]), dim=0) for i in range(len(original_states))), dim=0)
        cons_key_value_tensor = torch.stack(list(torch.stack(list(cons_states[i]), dim=0) for i in range(len(cons_states))), dim=0)
        # If root, store the whole states
        if(node_index == 0):
            for b in range(len(original_tokens_ids)):
                for i, label_name in enumerate(label_names):
                    self._states[label_name][(b, node_index)] = torch.clone(cons_key_value_tensor[:, :, i*len(original_tokens_ids)+b])
                self._states["original"][(b, node_index)] = torch.clone(original_key_value_tensor[:, :, b])
        # Else just store the additional token hidden states to save space
        else:
            for b in range(len(original_tokens_ids)):
                for i, label_name in enumerate(label_names):
                    self._states[label_name][(b, node_index)] = torch.clone(cons_key_value_tensor[:, :, i*len(original_tokens_ids)+b, :, -1:])
                self._states["original"][(b, node_index)] = torch.clone(original_key_value_tensor[:, :, b, :, -1:])
                

        # Updates tokens ids 
        for b, original_token_ids in enumerate(original_tokens_ids):
            for i, label_name in enumerate(label_names):
                self._token_ids[label_name][(b, node_index)] = cons_input_ids[label_name][b]
            self._token_ids["original"][(b, node_index)] = original_token_ids
            
        # Updates attention masks
        for b, original_attention_mask in enumerate(original_attention_masks):
            for i, label_name in enumerate(label_names):
                self._attention_mask[label_name][(b, node_index)] = cons_attention_mask[label_name][b]
            self._attention_mask["original"][(b, node_index)] = original_attention_mask

    def backward(self, leaf_indices):
        """Goes up and updates the tree until all nodes reached the root."""
        node_indices = leaf_indices # (B)
        leaf_values = self._values[self._batch_range, leaf_indices]
        while True:
            is_root = node_indices == 0
            if is_root.all():
                return
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            root_mask = 1.0 * is_root
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            # Update the parent nodes iff their child is not the root.
            # We therefore mask the updates using not_root_mask and root_mask.
            self._values[self._batch_range, parents] = not_root_mask * (self._values[self._batch_range, parents] *
                self._visit_counts[self._batch_range, parents] + leaf_values) / (self._visit_counts[self._batch_range,
                parents] + 1.0) + root_mask * self._values[self._batch_range, parents]

            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            self._children_values[self._batch_range, parents, actions] = not_root_mask * self._values[self._batch_range,node_indices] + root_mask * self._children_values[self._batch_range, parents, actions]
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            # Go up
            node_indices = parents

def main():
    print("loading dataset")
    data_lines = pd.read_csv("datasets/ag_news/full/prompts.tsv", sep='\t', engine='python', encoding="utf8")
    print("dataset loaded")
    generated_counter = 0
    samples_size = 501
    batch_size = 25
    labels = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    texts = {}
    for label_name in label_names:
        texts[label_name] = [None] * batch_size
    texts["original"] = [None] * batch_size
    MCTS = NumpyMCTS(root_fun, rec_fun, batch_size=batch_size, num_simulations=args.num_it, num_actions=vocab_size+1, num_sparse_actions=50, pb_c_init=args.c, temperature = args.temperature, alpha=args.alpha, penalty=args.penalty)
    samples_pbar = tqdm(total = samples_size, desc="Samples generated")
    while(generated_counter + batch_size <= samples_size): 
        labels.fill_(0)
        # Prepare search inputs
        lines = data_lines[generated_counter:generated_counter+batch_size]
    
        
        for i, (_, row) in enumerate(lines.iterrows()):
            labels[i] = int(row["label"])
            for label_name in label_names:
                texts[label_name][i] = "[CLS] [" + label_name.upper() + "] " + str(row["text"]) 
            texts["original"][i] = "[CLS] "+ str(row["text"]) 
          
        MCTS.set_labels(labels)
        original_input = tokenizer(texts["original"], return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=20).to("cuda")
        cons_input_ids = {}
        cons_attention_masks = {}
        for label_name in label_names:
            tokenizer_res = tokenizer(texts[label_name], return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=24).to("cuda")
            cons_input_ids[label_name] = tokenizer_res["input_ids"]
            cons_attention_masks[label_name] = tokenizer_res["attention_mask"]
        
        MCTS.search(original_input, cons_input_ids, cons_attention_masks)
        generated_counter += batch_size
        samples_pbar.update(batch_size)
            



if __name__ == "__main__":
    main()