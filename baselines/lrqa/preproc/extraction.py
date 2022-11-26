import os
from typing import Iterable
import torch
from rouge_score import rouge_scorer
import spacy
from tqdm import tqdm
import pandas as pd
import this_io
import json
from bs4 import BeautifulSoup
import sys
sys.path.append('/home/yuanhuang/quality2/quality/baselines')
import numpy as np
import nltk
import transformers
import lrqa.preproc.simple as simple
import torch.nn.functional as F


class SimpleScorer:
    def __init__(self, metrics=(("rouge1","r"),), use_stemmer=True):
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(
            [metric[0] for metric in self.metrics],
            use_stemmer=use_stemmer,
        )

    def score(self, reference: str, target: str):
        scores = self.scorer.score(reference, target)
        sub_scores = []
        for metric, which_score in self.metrics:
            score = scores[metric]
            if which_score == "p":
                score_value = score.precision
            elif which_score == "r":
                score_value = score.recall
            elif which_score == "f":
                score_value = score.fmeasure
            else:
                raise KeyError(which_score)
            sub_scores.append(score_value)
        return np.mean(sub_scores)


class FastTextScorer:
    def __init__(self, data, use_cache=True, verbose=True):
        if isinstance(data, str):
            data = torch.load(data)
        self.data_dict = {k: data["arr_data"][i] for i, k in enumerate(data["keys"])}
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', "lemmatizer", "attribute_ruler"])
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
        self.verbose = verbose
        self.unk_set = set()

    def _embed_single(self, string: str):
        token_list = [str(token) for token in self.nlp(string)]
        token_embeds = []
        for token in token_list:
            if token in self.data_dict:
                token_embeds.append(self.data_dict[token])
            else:
                if self.verbose and token not in self.unk_set:
                    print(f"Verbose: Did not find '{token}'")
                    self.unk_set.add(token)
        if not token_embeds:
            return np.zeros(300)
        token_embeds = np.array(token_embeds)
        return token_embeds.mean(0)

    def score(self, reference: str, target: str):
        if self.use_cache:
            if reference not in self.cache:
                self.cache[reference] = self._embed_single(reference)
            if target not in self.cache:
                self.cache[target] = self._embed_single(target)
            ref_embed = self.cache[reference]
            tgt_embed = self.cache[target]
        else:
            ref_embed = self._embed_single(reference)
            tgt_embed = self._embed_single(target)
        return cosine_similarity(ref_embed, tgt_embed)


class DPRScorer:
    def __init__(self,
                 context_encoder_name="facebook/dpr-ctx_encoder-multiset-base",
                 question_encoder_name="facebook/dpr-question_encoder-multiset-base",
                 device=None,
                 use_cache=True, verbose=True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
        self.context_encoder = transformers.DPRContextEncoder.from_pretrained(context_encoder_name).to(device)
        self.question_encoder = transformers.DPRQuestionEncoder.from_pretrained(question_encoder_name).to(device)
        self.device = device
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
        self.verbose = verbose
        self.unk_set = set()

    def _convert_to_batch(self, string):
        return {k: torch.tensor([v]).to(self.device) for k, v in self.tokenizer(string).items()}

    def _embed_context(self, context: str):
        context_batch = self._convert_to_batch(context)
        with torch.no_grad():
            out = self.context_encoder(**context_batch)
        return out.pooler_output[0].cpu().numpy()

    def _embed_question(self, question: str):
        query_batch = self._convert_to_batch(question)
        with torch.no_grad():
            out = self.question_encoder(**query_batch)
        return out.pooler_output[0].cpu().numpy()

    def score(self, reference: str, target: str):
        # Reference <- question
        # Target <- context
        if self.use_cache:
            if reference not in self.cache:
                self.cache[reference] = self._embed_question(reference)
            if target not in self.cache:
                self.cache[target] = self._embed_context(target)
            ref_embed = self.cache[reference]
            tgt_embed = self.cache[target]
        else:
            ref_embed = self._embed_question(reference)
            tgt_embed = self._embed_context(target)
        return -np.linalg.norm(ref_embed - tgt_embed)


def cosine_similarity(arr1, arr2):
    return F.cosine_similarity(
        torch.from_numpy(arr1.reshape(1, 300)),
        torch.from_numpy(arr2.reshape(1, 300)),
    )[0]


def get_sent_data(raw_text, clean_text=True, sliding_window = False, window_size = 100):
    """Given a passage, return sentences and word counts."""
    max_chunk_word = window_size
    sent_data_sliding_window = []
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    if clean_text:
        if isinstance(raw_text, list):
            raw_text = "\n".join(raw_text)
        context = simple.format_nice_text(raw_text)
    else:
        assert isinstance(raw_text, str)
        context = raw_text
    sent_data = []
    ind = 0
    for sent_obj in nlp(context).sents:
        sent_data.append({
            "index": ind,
            "text": str(sent_obj).strip(),
            "word_count": len(sent_obj),
        })
        ind = ind + 1
    if sliding_window:
        #each window [i,j]
        i = 0
        while i < len(sent_data):
            j = i
            chunk_word_count = 0
            while j < len(sent_data) and chunk_word_count + sent_data[j]["word_count"] < max_chunk_word :
                chunk_word_count += sent_data[j]["word_count"]
                j += 1
            sent_data_sliding_window.append({})
            sent_data_sliding_window[-1]["index"] = i
            text = ""
            for txt in range(i, j):  # beacause j has already added 1
                text = text + sent_data[txt]["text"]
            span = []
            for n in range(i, j):
                span.append(n)
            sent_data_sliding_window[-1]["text"] = text
            sent_data_sliding_window[-1]["word_count"] = chunk_word_count
            sent_data_sliding_window[-1]["span"] = span
            i += 1
        return sent_data_sliding_window
    return sent_data


def get_top_sentences(query: str, sent_data: list, max_word_count: int, scorer: SimpleScorer, smooth = False, M = 9, smooth_mean = False):
    scores = []#added
    score = []

    if smooth:
        for sent_idx, sent_dict in enumerate(sent_data):
            score.append(float(scorer.score(query, sent_dict["text"])))
            #scores.append((sent_idx, float(scorer.score(query, sent_dict["text"]))))
        #import ipdb;ipdb.set_trace()
        if smooth_mean:
            score_mean = np.mean(score)
            for i in range(len(score)):
                if score[i] < score_mean:
                    score[i] = score_mean
        score_smooth = smooth2nd(score,M)
        for id in range(len(sent_data)):
            scores.append((id,score_smooth[id]))
    else:
        for sent_idx, sent_dict in enumerate(sent_data):
            score.append(float(scorer.score(query, sent_dict["text"])))
            scores.append((sent_idx, float(scorer.score(query, sent_dict["text"]))))
    # Sort by score, in descending order
    sorted_scores = sorted(scores, key=lambda _: _[1], reverse=True)
    selected_span = [] # get rid of repeated chunks
    # Choose highest scoring sentences
    chosen_sent_indices = []
    total_word_count = 0
    for sent_idx, score in sorted_scores:
        span = 'span'
        if span in sent_data[sent_idx].keys():
            if sent_data[sent_idx]["span"][0] in selected_span:
                continue
            sent_word_count = sent_data[sent_idx]["word_count"]
            if total_word_count + sent_word_count > max_word_count:
                break
            chosen_sent_indices.append(sent_idx)
            for i in sent_data[sent_idx]["span"]:
                selected_span.append(i)
            total_word_count += sent_word_count
        else:
            sent_word_count = sent_data[sent_idx]["word_count"]
            if total_word_count + sent_word_count > max_word_count:
                break
            chosen_sent_indices.append(sent_idx)
            total_word_count += sent_word_count

    #

    # Re-condense article
    shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in sorted(chosen_sent_indices))
    return shortened_article,sorted_scores,chosen_sent_indices

def smooth2nd(x,M):
    K = round(M/2-0.1)
    lenX = len(x)
    y = np.zeros(lenX)
    for NN in range(0,lenX,1):
        startInd = max([0,NN-K])
        endInd = min(NN+K+1,lenX)
        y[NN] = np.mean(x[startInd:endInd])
    return y

def format_nice_text(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    p_list = soup.findAll('p')
    if len(p_list) == 0:
        # Fall-back for if we have no <p> tags to work off
        return " ".join(soup.get_text().strip().split())
    else:
        text_list = []
        header = get_clean_text(p_list[0].prev_sibling)
        if header:
            text_list.append(header)
        for p_elem in p_list:
            clean_p_text = get_clean_text(p_elem.get_text())
            if clean_p_text:
                text_list.append(clean_p_text)
            clean_p_suffix = get_clean_text(p_elem.next_sibling)
            if clean_p_suffix:
                text_list.append(clean_p_suffix)
        return "\n\n".join(text_list)

def process_file_long(input_path, output_path, strip_html=False):
    data = this_io.read_jsonl(input_path)
    out = []
    for row in data:
        i = 1
        if strip_html:
            context = format_nice_text("\n\n".join(row["article"]))
        else:
            context = row["article"]
        while True:
            if f"question{i}" not in row:
                break
            out.append({
                "context": "".join(context),
                "query": " " + row[f"question{i}"].strip(),
                "option_0": " " + row[f"question{i}option1"].strip(),
                "option_1": " " + row[f"question{i}option2"].strip(),
                "option_2": " " + row[f"question{i}option3"].strip(),
                "option_3": " " + row[f"question{i}option4"].strip(),
                "label": row[f"question{i}_gold_label"] - 1,
            })
            i += 1
    this_io.write_jsonl(out, output_path)

def process_file(input_path, output_path, phase, args, scorer: SimpleScorer, query_type="question", max_word_count=300,
                 verbose=False, clean_text=True,smooth=False,M=8,smooth_mean=False):
    data = this_io.read_jsonl(input_path)
    out = []
    dpr_score = []
    all_chosen_indices = []
    all_sent_data = []
#    for row in tqdm.tqdm(data, verbose=verbose):
#    for row in tqdm.tqdm(data): type of data:list
    for row in tqdm(data):
        sent_data = get_sent_data(row["article"], clean_text=clean_text, sliding_window=args.sliding_window,window_size = args.window_size)
        all_sent_data.append(sent_data)
        i = 1
        chosen_indices = set()
        while True:
            if f"question{i}" not in row:
                break
            if query_type == "question":
                query = row[f"question{i}"].strip()
            elif query_type == "oracle_answer":
                query = row[f"question{i}option{row[f'question{i}_gold_label']}"].strip()
            elif query_type == "oracle_question_answer":
                query = (
                    row[f"question{i}"].strip()
                    + " " + row[f"question{i}option{row[f'question{i}_gold_label']}"].strip()
                )
            elif query_type == "all_answers":
                opt = 1
                query_str = row[f"question{i}option{opt}"].strip()

                while (f"question{i}option{opt+1}" in row.keys()):
                    opt = opt + 1
                    query_str = query_str + " " + row[f"question{i}option{opt}"].strip()
                query = (
                    query_str
                )
            elif query_type == "question_all_answers":
                opt = 1
                query_str = row[f"question{i}"].strip()+ " " + row[f'question{i}option{opt}'].strip()

                while (f"question{i}option{opt+1}" in row.keys()):
                    opt = opt + 1
                    query_str = query_str + " " + row[f"question{i}option{opt}"].strip()
                query = (
                    query_str
                )
            elif query_type == "separate_answers":

                SIDE_OPTION_WORD_COUNT = 300
                GOLD_OPTION_WORD_COUNT = 300
                query_str = ""
                opt = 1
                while (f"question{i}option{opt}" in row.keys()):
                    cur_query = (row[f"question{i}option{opt}"])
                    if opt == row[f"question{i}_gold_label"]:
                        #import ipdb;ipdb.set_trace()
                        cur_shortened_article, sorted_score,cur_indices = get_top_sentences(
                            query=cur_query,
                            sent_data=sent_data,
                            max_word_count=SIDE_OPTION_WORD_COUNT,
                            scorer=scorer,
                            smooth=smooth,
                            M = M,
                            smooth_mean=smooth_mean,
                        )
                    else:
                        cur_shortened_article, sorted_score,cur_indices = get_top_sentences(
                            query=cur_query,
                            sent_data=sent_data,
                            max_word_count=GOLD_OPTION_WORD_COUNT,
                            scorer=scorer,
                            smooth=smooth,
                            M=M,
                            smooth_mean=smooth_mean,
                        )
                    opt += 1
                    for ind in cur_indices:
                        chosen_indices.add(ind)


                opt = 1
                query_str = row[f"question{i}option{opt}"].strip()

                while (f"question{i}option{opt + 1}" in row.keys()):
                    opt = opt + 1
                    query_str = query_str + " " + row[f"question{i}option{opt}"].strip()
                query = (
                    query_str
                )

            else:
                raise KeyError(query_type)

            if query_type == "separate_answers":

                #import ipdb;ipdb.set_trace()
                cur_shortened_article = []
                for ind in sorted(list(chosen_indices)):
                    cur_shortened_article.append(sent_data[ind])
                shortened_article, sorted_score, cur_indices = get_top_sentences(
                    query=query,
                    sent_data=cur_shortened_article,
                    max_word_count=max_word_count,
                    scorer=scorer,
                    smooth=smooth,
                    M=M,
                    smooth_mean=smooth_mean,
                )
            else:


                shortened_article, sorted_score, chosen_indices = get_top_sentences(
                    query=query,
                    sent_data=sent_data,
                    max_word_count=max_word_count,  # max 300
                    scorer=scorer,
                    smooth=smooth,
                    M=M,
                    smooth_mean=smooth_mean,
                )
                dpr_score.append(sorted_score)
                all_chosen_indices.append(chosen_indices)
            out.append({
                "context": shortened_article,
                "query": " " + row[f"question{i}"].strip(),
                "this_query": query,
                "option_0": " " + row[f"question{i}option1"].strip(),
                "option_1": " " + row[f"question{i}option2"].strip(),
                "option_2": " " + row[f"question{i}option3"].strip(),
                "option_3": " " + row[f"question{i}option4"].strip(),
                "label": row[f"question{i}_gold_label"] - 1,
            })
            i += 1
    output_phase_path = os.path.join(output_path, f"{phase}.jsonl")
    sent_data_path = os.path.join(output_path, f"sent_data_{phase}.json")
    score_path = os.path.join(output_path, f"score_{phase}.json")
    chosen_indices_path = os.path.join(output_path, f"chosen_indices_{phase}.json")

    this_io.write_jsonl(out, output_phase_path)
    this_io.write_jsonl(all_sent_data, sent_data_path)

    #import ipdb;ipdb.set_trace()
    json.dump(dpr_score,open(score_path, 'w'),
              sort_keys=True, indent=4, separators=(',', ': '))
    json.dump(all_chosen_indices, open(chosen_indices_path, 'w'),
              sort_keys=True, indent=4, separators=(',', ': '))
