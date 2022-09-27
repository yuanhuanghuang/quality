import json
import os
import pandas as pd
import extraction
import preprocess_html as pre
import this_io
def f():
    input_path = '../../../data/v1.0.1/QuALITY.v1.0.1.train'
    output_dir = '../jsonFile'
    output_filename = 'train.json'
    output_file = os.path.join(output_dir,output_filename)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    data = pd.read_json(input_path,lines = True)
    col = ['article','questions']
    newData = pd.DataFrame(columns = col)
    for i in col:
        newData[i] = data[i]
    all_data = []
    for i in range(data.shape[0]):
        cur_data = {}
        cur_data['article'] = pre.strip_html(data.iloc[i]['article']).decode()

        for j in range(1,len(data.iloc[i]['questions'])+1):
            cur_data[f"question{j}"] = data.iloc[i]['questions'][j-1]['question']
            for q in range(1,5):
                cur_data[f"question{j}option{q}"] = data.iloc[i]['questions'][j - 1]['options'][q-1]
            cur_data[f"question{j}_gold_label"] = data.iloc[i]['questions'][j - 1]['gold_label']
        all_data.append(cur_data)
    output_inter_file = 'inter_train.json'
    output_inter_file = os.path.join(output_dir, output_inter_file)
    this_io.write_jsonl(all_data,output_inter_file)
    # extraction
    input_path = output_inter_file
    output_path = output_file
    extraction.process_file(input_path, output_path, scorer=extraction.SimpleScorer(), query_type="question", max_word_count=300,
                            verbose=False, clean_text=True)


f()