import argparse
import json
import os
import pandas as pd
import extraction
import preprocess_html as pre
import this_io
PHASES = ["train", "validation", "test"]
def f(input_path):
    temp_phase = ["train", "dev", "test"]
    for temp in range(len(temp_phase)):
        file_name = 'QuALITY.v1.0.1.htmlstripped.' + temp_phase[temp]
        input_file = os.path.join(input_path, file_name)
        data = pd.read_json(input_file,lines = True)
        col = ['article','questions']
        newData = pd.DataFrame(columns = col)
        for i in col:
            newData[i] = data[i]
        all_data = []
        for i in range(data.shape[0]):
            cur_data = {}
            #cur_data['article'] = pre.strip_html(data.iloc[i]['article']).decode()
            cur_data['article'] = data.iloc[i]['article']
            for j in range(1,len(data.iloc[i]['questions'])+1):
                cur_data[f"question{j}"] = data.iloc[i]['questions'][j-1]['question']
                for q in range(1,5):
                    cur_data[f"question{j}option{q}"] = data.iloc[i]['questions'][j - 1]['options'][q-1]
                if temp_phase[temp]!="test":
                    cur_data[f"question{j}_gold_label"] = data.iloc[i]['questions'][j - 1]['gold_label']
            all_data.append(cur_data)
        output_inter_file = PHASES[temp] + '.jsonl'
        #output_inter_file = 'train_format/' + output_inter_file
        output_inter_file = os.path.join(input_path, output_inter_file)
        this_io.write_jsonl(all_data,output_inter_file)

def get_scorer(scorer_name, args):
        if scorer_name == "rouge":
            return extraction.SimpleScorer()
        elif scorer_name == "fasttext":
            return extraction.FastTextScorer(extraction.load_fasttext_vectors(
                fname=args.fasttext_path,
                max_lines=100_000,
                ))
        elif scorer_name == "dpr":
            return extraction.DPRScorer(device="cuda:0")
        else:
            raise KeyError(scorer_name)
def main():
    parser = argparse.ArgumentParser(description="Do extractive preprocessing")
    parser.add_argument("--input_base_path", type=str, required=True,
            help="Path to folder of cleaned inputs")
    parser.add_argument("--output_base_path", type=str, required=True,
            help="Path to write processed outputs to")
    parser.add_argument("--scorer", type=str, default="rouge",
            help="{rouge, fasttext, dpr}")
    parser.add_argument("--query_type", type=str, default="question",
            help="{question, oracle_answer, oracle_question_answer}")
    parser.add_argument("--fasttext_path", type=str, default="/path/to/crawl-300d-2M.vec",
            help="Pickle of fasttext vectors. (Only used for fasttext.)")
    parser.add_argument("--smooth", action='store_true', default=False,
                        help="If use smooth")
    parser.add_argument("--M", type=int, default=8,
                        help="smooth argument")
    parser.add_argument("--smooth_mean", action='store_true', default=False,
                        help="If use smooth mean method")
    parser.add_argument("--sliding_window", action='store_true', default=False,
                        help="If do sliding window (default <100 words)")
    parser.add_argument("--window_size", type=int, default=100,
                        help="smooth argument")
    parser.add_argument("--max_word_count", type=int, default=300,
                        help="chunk size")
    parser.add_argument("--longformer", action='store_true', default=False,
                        help="If use smooth mean method")
    args = parser.parse_args()
    os.makedirs(args.output_base_path, exist_ok=True)
    scorer = get_scorer(scorer_name=args.scorer, args=args)
    with open(os.path.join(args.output_base_path, 'training_args.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)

    for phase in PHASES:
        f(args.input_base_path)
        if (args.longformer):
            extraction.process_file_long(
                input_path=os.path.join(args.input_base_path, f"{phase}.jsonl"),
                output_path=os.path.join(args.output_base_path, f"{phase}.jsonl"),
            )
        else:
            extraction.process_file(
                    input_path=os.path.join(args.input_base_path, f"{phase}.jsonl"),
                    output_path=args.output_base_path,
                    phase = phase,
                    args=args,
                    scorer=scorer,
                    query_type=args.query_type,
                    max_word_count = args.max_word_count,
                    smooth=args.smooth,
                    M=args.M,
                    smooth_mean=args.smooth_mean,
                    )


if __name__ =='__main__':
    main()

