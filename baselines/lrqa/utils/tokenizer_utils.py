import numpy as np

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing import Dict

import lrqa.tasks as tasks
import torch
import re
from torch import tensor

def tokenize_examples_for_enc_dec_model(examples, tokenizer, max_seq_length: int,
                                        padding_strategy: PaddingStrategy,
                                        truncation_strategy: TruncationStrategy):
    option_keys = sorted([
        key for key in examples
        if key.startswith("option_")
    ])
    input_strs = []
    target_strs = []
    for i in range(len(examples[option_keys[0]])):
        all_options = " ".join([f"choice {j}: {examples[option_key][i]}" for j, option_key in enumerate(option_keys)])
        input_str = f" context: {examples['context'][i]} </s> {all_options} question: {examples['query'][i]} </s>"
        target_str = f"{examples['label'][i]}"
        input_strs.append(input_str)
        target_strs.append(target_str)
        
    tokenized_inputs = tokenizer(
        input_strs,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        target_strs,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )

    target_ids = tokenized_targets["input_ids"]
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized_inputs["input_ids"].numpy(),
        "attention_mask": tokenized_inputs["attention_mask"].numpy(),
        "labels": target_ids.numpy(),
    }


def tokenize_examples_for_all_opts_mc_model(examples, tokenizer, max_seq_length: int,
                                        padding_strategy: PaddingStrategy,
                                        truncation_strategy: TruncationStrategy):
    option_keys = sorted([
        key for key in examples
        if key.startswith("option_")
    ])
    input_strs = []
    target_strs = []
    eos = '<eos>'
    eos_token = tokenizer(eos)['input_ids'][1]
    for i in range(len(examples[option_keys[0]])):
        #There are all 6 <eos>
        #I can set only 4 <eos>
        all_options = "".join([f" <eos> choice {j}: { examples[option_key][i] }" for j, option_key in enumerate(option_keys)])
        input_str = f"{all_options} question: {examples['query'][i]}  context: {examples['context'][i]}"
        target_str = f"{examples['label'][i]}"
        input_strs.append(input_str)
        target_strs.append(target_str)


    tokenized_inputs = tokenizer(
        input_strs,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )

    index_map = []
    for i in range(len(tokenized_inputs['input_ids'])):
        this_map=[]
        for ind in range(len(tokenized_inputs['input_ids'][i].tolist())):
            if tokenized_inputs['input_ids'][i][ind] == eos_token:
                this_map.append(ind)
        index_map.append(this_map)
    index_map = torch.tensor(index_map)

    tokenized_targets = tokenizer(
        target_strs,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )
    target_ids = tokenized_targets["input_ids"]
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized_inputs["input_ids"].numpy(),
        "attention_mask": tokenized_inputs["attention_mask"].numpy(),
        "labels": target_ids.numpy(),
        'index_map': index_map.numpy(),
    }


def tokenize_examples_for_mc_lm_model(examples, tokenizer, max_seq_length: int,
                                      padding_strategy: PaddingStrategy,
                                      truncation_strategy: TruncationStrategy):
    """
    Takes a dictionary of examples, with keys:
        context: str (before [SEP])
        query: str (after [SEP], can be empty)
        option_0: str
        option_1: str
        ...
        label: int
    """
    # This assumes option_keys sorted order corresponds labels order
    # which is fine for num_labels < 10
    option_keys = sorted([
        key for key in examples
        if key.startswith("option_")
    ])
    result = {
        "label": examples["label"],
    }
    for option_key in option_keys:
        input_part2 = [
            query + option
            for query, option
            in zip(examples["query"], examples[option_key])
        ]
        tokenized_option = tokenizer(
            examples["context"],
            input_part2,
            padding=padding_strategy,
            max_length=max_seq_length,
            truncation=truncation_strategy,
        )

        # For generation
        option_token_end_idx = np.array(tokenized_option["attention_mask"]).sum(-1)
        # heuristic, because tokenizers can be weird
        option_token_start_idx = option_token_end_idx - np.array([
            len(tokenizer.tokenize(x))
            for x in examples[option_key]
        ])
        # noinspection PyUnresolvedReferences
        assert (option_token_start_idx < option_token_end_idx).all()
        tokenized_option["option_token_start_idx"] = option_token_start_idx
        tokenized_option["option_token_end_idx"] = option_token_end_idx

        # Append to option lists
        for k, v in tokenized_option.items():
            if k not in result:
                result[k] = [[v_elem] for v_elem in v]
            else:
                for i, v_elem in enumerate(v):
                    result[k][i].append(v_elem)

    return result


def get_tokenized_dataset(task: tasks.Task, dataset_dict,
                          tokenizer,
                          max_seq_length: int,
                          padding_strategy: PaddingStrategy,
                          truncation_strategy: TruncationStrategy,
                          model_mode: str,
                          model_path :str,

                          ) -> Dict:
    tokenized_dataset = {}
    for phase in ["train", "validation", "test"]:
        if phase not in dataset_dict:
            continue
        standard_examples = dataset_dict[phase].map(
            task.standardize_examples,
            batched=True,
            remove_columns=task.drop_columns,
        )
        if model_mode in ["mc", "generation"]:
            if 'deberta' in model_path :
                tokenize_examples = lambda examples: tokenize_examples_for_all_opts_mc_model(examples, tokenizer,
                                                                                       max_seq_length,
                                                                                       padding_strategy,
                                                                                       truncation_strategy)
            else:
                tokenize_examples = lambda examples: tokenize_examples_for_mc_lm_model(examples, tokenizer, max_seq_length,
                                                                                       padding_strategy,
                                                                                       truncation_strategy)
        else:
            tokenize_examples = lambda examples: tokenize_examples_for_enc_dec_model(examples, tokenizer,
                                                                                     max_seq_length,
                                                                                     padding_strategy,
                                                                                     truncation_strategy)
        tokenized_examples = standard_examples.map(tokenize_examples, batched=True)
        '''
        # for mc model
        Dataset(features: {'attention_mask': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None), 
        'context': Value(dtype='string', id=None), 
        'input_ids': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None), 
        'label': Value(dtype='int64', id=None), 'option_0': Value(dtype='string', id=None), 
        'option_1': Value(dtype='string', id=None), 
        'option_2': Value(dtype='string', id=None), 
        'option_3': Value(dtype='string', id=None), 
        'option_token_end_idx': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), 
        'option_token_start_idx': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), 
        'query': Value(dtype='string', id=None), 'this_query': Value(dtype='string', id=None), 
        'token_type_ids': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None)}, 
        num_rows: 2523)
        '''
        tokenized_dataset[phase] = tokenized_examples
    return tokenized_dataset
