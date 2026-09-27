"""
Output Generation Script for Business Entity Resolution Challenge.

Generates:
1. output/matching_results.tsv
2. output/candidate_pairs.tsv

And validates both against student_resource/utils/validate_submission.py rules.
"""

import os
import sys
import argparse
import subprocess

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from src.infer import run_inference
except ImportError:
    from infer import run_inference


def main():
    parser = argparse.ArgumentParser(description="Generate submission TSV files.")
    parser.add_argument("--test-s1", default="student_resource/dataset/test/test_source1.tsv", help="Path to test S1 TSV")
    parser.add_argument("--test-s2", default="student_resource/dataset/test/test_source2.tsv", help="Path to test S2 TSV")
    parser.add_argument("--test-s3", default="student_resource/dataset/test/test_source3.tsv", help="Path to test S3 TSV")
    parser.add_argument("--output-dir", default="output", help="Directory to save TSV outputs")
    parser.add_argument("--model-path", default="models/lgbm_matcher.pkl", help="Trained model path")
    parser.add_argument("--config-path", default="models/config.json", help="Model config path")
    parser.add_argument("--threshold", type=float, default=None, help="Optional manual threshold override")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("==================================================")
    print("        RESOLVEX: PIPELINE EXECUTION              ")
    print("==================================================")

    run_inference(
        s1_path=args.test_s1,
        s2_path=args.test_s2,
        s3_path=args.test_s3,
        model_path=args.model_path,
        config_path=args.config_path,
        override_threshold=args.threshold,
        output_dir=args.output_dir
    )

    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    # Run submission validator
    validator_script = "student_resource/utils/validate_submission.py"
    if os.path.exists(validator_script):
        print("\n>>> Running Challenge Submission Validator...")
        test_dir = os.path.dirname(args.test_s1)
        cmd = [
            sys.executable,
            validator_script,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            if res.returncode == 0:
                print("[SUCCESS] Output files successfully passed all challenge constraints!")
            else:
                print(f"[STATUS] Validator exit code: {res.returncode}")
        except Exception as e:
            print(f"Error running validator: {e}")
    else:
        print(f"Note: Validator not found at {validator_script}, skipping automated check.")


if __name__ == '__main__':
    main()
