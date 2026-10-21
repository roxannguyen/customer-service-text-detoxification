import os
import pandas as pd

# Paths relative to 04_evaluation
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

DATA_PROCESSED = os.path.join(BASE_DIR, 'data_processed')
DATA_RECEIVED = os.path.join(BASE_DIR, 'data_received')
DATA_GOLD = os.path.join(BASE_DIR, 'data_human_gold_label')
os.makedirs(DATA_GOLD, exist_ok=True)
# Output directly to BASE_DIR

def build_task_a_gold():
    path_a01 = os.path.join(DATA_RECEIVED, 'task_a/task_a_A01.csv')
    path_a02 = os.path.join(DATA_RECEIVED, 'task_a/task_a_A02.csv')
    path_dis = os.path.join(DATA_RECEIVED, 'task_a/disagreements_task_a.csv')
    path_meta = os.path.join(DATA_PROCESSED, 'task_a/task_a_web_annotation_240.csv')

    df_a01 = pd.read_csv(path_a01, dtype=str, keep_default_na=False)
    df_a02 = pd.read_csv(path_a02, dtype=str, keep_default_na=False)
    df_dis = pd.read_csv(path_dis, dtype=str, keep_default_na=False)
    df_meta = pd.read_csv(path_meta, dtype=str, keep_default_na=False)

    merged = pd.merge(
        df_a01[['sample_id', 'human_toxicity_label']],
        df_a02[['sample_id', 'human_toxicity_label']],
        on='sample_id',
        suffixes=('_a01', '_a02')
    )

    dis_map = dict(zip(df_dis['sample_id'], df_dis['A03_label']))

    gold_labels = []
    a03_applied = []
    for _, row in merged.iterrows():
        sid = row['sample_id']
        l1, l2 = row['human_toxicity_label_a01'], row['human_toxicity_label_a02']
        if l1 == l2:
            gold_labels.append(l1)
            a03_applied.append('')
        else:
            gold_labels.append(dis_map.get(sid, ''))
            a03_applied.append(dis_map.get(sid, ''))

    merged['labelA01'] = merged['human_toxicity_label_a01']
    merged['labelA02'] = merged['human_toxicity_label_a02']
    merged['labelA03'] = a03_applied
    merged['gold_label'] = gold_labels

    out_df = pd.merge(merged[['sample_id', 'labelA01', 'labelA02', 'labelA03', 'gold_label']], df_meta, left_on='sample_id', right_on='id', how='left')
    out_path = os.path.join(DATA_GOLD, 'task_a_goldlabel_240.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8')
    print('Generated Task A Gold:', out_path, out_df.shape)

def build_task_b_gold():
    path_a01 = os.path.join(DATA_RECEIVED, 'task_b/task_b_A01.csv')
    path_a02 = os.path.join(DATA_RECEIVED, 'task_b/task_b_A02.csv')
    path_dis = os.path.join(DATA_RECEIVED, 'task_b/disagreements_task_b.csv')
    path_map = os.path.join(DATA_PROCESSED, 'task_b/task_b_pair_mapping_300.csv')

    df_a01 = pd.read_csv(path_a01, dtype=str, keep_default_na=False).set_index('sample_id')
    df_a02 = pd.read_csv(path_a02, dtype=str, keep_default_na=False).set_index('sample_id')
    df_dis = pd.read_csv(path_dis, dtype=str, keep_default_na=False).set_index('sample_id')
    df_map = pd.read_csv(path_map, dtype=str, keep_default_na=False)

    sample_ids = df_a01.index.tolist()
    records = []

    for sid in sample_ids:
        row1 = df_a01.loc[sid]
        row2 = df_a02.loc[sid]
        is_dis = sid in df_dis.index
        row_dis = df_dis.loc[sid] if is_dis else None

        # Resolve 4 criteria
        # 1. Toxicity
        if row1['output_toxicity'] == row2['output_toxicity']:
            g_tox = row1['output_toxicity']
        else:
            g_tox = row_dis['A03_toxicity'] if row_dis is not None and row_dis['A03_toxicity'] != '' else row1['output_toxicity']

        # 2. Content
        if row1['content_preservation'] == row2['content_preservation']:
            g_cont = row1['content_preservation']
        else:
            g_cont = row_dis['A03_content'] if row_dis is not None and row_dis['A03_content'] != '' else row1['content_preservation']

        # 3. Naturalness
        if row1['naturalness'] == row2['naturalness']:
            g_nat = row1['naturalness']
        else:
            g_nat = row_dis['A03_naturalness'] if row_dis is not None and row_dis['A03_naturalness'] != '' else row1['naturalness']

        # 4. Request
        if row1['request_retention_human'] == row2['request_retention_human']:
            g_req = row1['request_retention_human']
        else:
            g_req = row_dis['A03_request'] if row_dis is not None and row_dis['A03_request'] != '' else row1['request_retention_human']

        records.append({
            'sample_id': sid,
            'input_text': row1['input_text'],
            'output_text': row1['output_text'],
            'system_id': row1.get('system_id', ''),
            'original_label': row1.get('original_label', ''),
            'band': row1.get('band', ''),
            'subtype': row1.get('subtype', ''),
            'gold_toxicity': g_tox,
            'gold_content': g_cont,
            'gold_naturalness': g_nat,
            'gold_request': g_req,
            'is_adjudicated': is_dis,
            'status': 'resolved'
        })

    gold_234 = pd.DataFrame(records)
    gold_234_path = os.path.join(DATA_GOLD, 'task_b_editbehavior_goldlabel_234.csv')
    gold_234.to_csv(gold_234_path, index=False, encoding='utf-8')
    print('Generated Task B 234 pairs gold:', gold_234_path, gold_234.shape)

    # Map to 300 outputs
    # Join pair mapping with gold labels on pair_id == sample_id
    gold_map_dict = gold_234.set_index('sample_id').to_dict(orient='index')
    records_300 = []
    for _, map_row in df_map.iterrows():
        pair_id = map_row['pair_id']
        g_info = gold_map_dict.get(pair_id, {})
        rec = dict(map_row)
        rec['gold_toxicity'] = g_info.get('gold_toxicity', '')
        rec['gold_content'] = g_info.get('gold_content', '')
        rec['gold_naturalness'] = g_info.get('gold_naturalness', '')
        rec['gold_request'] = g_info.get('gold_request', '')
        rec['is_adjudicated'] = g_info.get('is_adjudicated', False)
        records_300.append(rec)

    df_300 = pd.DataFrame(records_300)
    # Remove any Vietnamese translation columns if present
    for vi_col in ['output_text_vi', 'input_text_vi', 'vn_translate']:
        if vi_col in df_300.columns:
            df_300 = df_300.drop(columns=[vi_col])

    df_300_path = os.path.join(DATA_GOLD, 'task_b_system_outputs_300_goldlabel.csv')
    df_300.to_csv(df_300_path, index=False, encoding='utf-8')
    print('Generated Task B 300 outputs gold:', df_300_path, df_300.shape)

if __name__ == '__main__':
    build_task_a_gold()
    build_task_b_gold()
