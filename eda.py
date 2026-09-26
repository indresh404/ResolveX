import pandas as pd
import numpy as np

print("=== DATASET SHAPES AND DISTRIBUTIONS ===")
for split in ['train', 'test']:
    for src in ['source1', 'source2', 'source3']:
        fn = f'student_resource/dataset/{split}/{split}_{src}.tsv'
        df = pd.read_csv(fn, sep='\t')
        country_dist = df['country'].value_counts().to_dict()
        null_counts = df.isnull().sum().to_dict()
        print(f"[{split}_{src}] Shape: {df.shape} | Countries: {country_dist} | Nulls: {null_counts}")

gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
print(f"\n[train_ground_truth] Shape: {gt.shape} | Nulls: {gt.isnull().sum().to_dict()}")
gt['matches'] = gt['matched_entity_ids'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])
gt['match_count'] = gt['matches'].apply(len)
print("Match count distribution (matches per S1 entity):")
print(gt['match_count'].value_counts().sort_index())

# Check match breakdown by source
s2_matches = 0
s3_matches = 0
for m_list in gt['matches']:
    for m in m_list:
        if m.startswith('S2-'):
            s2_matches += 1
        elif m.startswith('S3-'):
            s3_matches += 1
print(f"Total Ground Truth Matches: S2 = {s2_matches}, S3 = {s3_matches}, Total = {s2_matches + s3_matches}")

# Inspect sample positive matches
print("\n=== SAMPLE MATCHES INSPECTION ===")
s1_df = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t').set_index('entity_id')
s2_df = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t').set_index('entity_id')
s3_df = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t').set_index('entity_id')

sample_gt = gt[gt['match_count'] >= 2].head(3)
for idx, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1_row = s1_df.loc[s1_id]
    print(f"\nS1 Entity [{s1_id}] ({s1_row['country']}):")
    print(f"  Name:    {s1_row['business_name']}")
    print(f"  Address: {s1_row['business_address']}")
    for m_id in row['matches']:
        if m_id in s2_df.index:
            m_row = s2_df.loc[m_id]
            src_name = "Source 2"
        elif m_id in s3_df.index:
            m_row = s3_df.loc[m_id]
            src_name = "Source 3"
        else:
            print(f"  Unknown match ID: {m_id}")
            continue
        print(f"  -> Match [{m_id}] ({src_name}, {m_row['country']}):")
        print(f"     Name:    {m_row['business_name']}")
        print(f"     Address: {m_row['business_address']}")
