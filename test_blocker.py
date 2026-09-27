import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.blocking import ScalableMultiStrategyBlocker
from src.evaluate import evaluate_blocking

print('Loading 20,000 training sample to evaluate blocking performance...')
s1_raw = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=20000)
s2_raw = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', nrows=50000)
s3_raw = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t', nrows=50000)
gt_df = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=20000)

s1_df = preprocess_dataframe(s1_raw, desc='S1')
s2_df = preprocess_dataframe(s2_raw, desc='S2')
s3_df = preprocess_dataframe(s3_raw, desc='S3')

blocker = ScalableMultiStrategyBlocker(max_candidates_per_s1=8)
cands = blocker.generate_candidates(s1_df, s2_df, s3_df)

stats = evaluate_blocking(gt_df, cands, len(s2_df) + len(s3_df))
print('\n=== BLOCKING BENCHMARK RESULTS ===')
print(f"Pair Completeness (Recall Ceiling): {stats['pair_completeness']*100:.2f}%")
print(f"Reduction Ratio:                    {stats['reduction_ratio']*100:.6f}%")
print(f"Avg Candidates / S1 Entity:         {stats['avg_candidates_per_s1']:.2f}")
