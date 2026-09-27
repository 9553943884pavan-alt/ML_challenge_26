#!/usr/bin/env python3
"""
aws_configure_and_launch.py
============================
Reads credentials from .env.aws and bootstraps the full pipeline.
Run this ONCE after filling in .env.aws with your keys.

Usage:
    python aws_configure_and_launch.py

    # To only download completed results (after pipeline runs):
    python aws_configure_and_launch.py --download-only

    # To monitor progress of a running pipeline:
    python aws_configure_and_launch.py --tail-only
"""

import os, sys, time, argparse, subprocess
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# ── Load credentials from .env.aws ───────────────────────────────────────────
env_file = BASE_DIR / ".env.aws"
if not env_file.exists():
    print(f"\n[ERROR] {env_file} not found!")
    print(f"\nPlease copy .env.aws.template → .env.aws and fill in your credentials:\n")
    print(f"  copy .env.aws.template .env.aws")
    print(f"  notepad .env.aws\n")
    sys.exit(1)

load_dotenv(env_file)

AWS_ACCESS_KEY_ID     = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_REGION            = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

if "REPLACE_WITH" in AWS_ACCESS_KEY_ID or not AWS_ACCESS_KEY_ID.strip():
    print("\n[ERROR] You haven't filled in your credentials in .env.aws yet!")
    print("Open .env.aws and replace the placeholder values.\n")
    sys.exit(1)

print(f"Credentials loaded from {env_file}")
print(f"Region: {AWS_REGION}")
print(f"Key ID: {AWS_ACCESS_KEY_ID[:4]}...{AWS_ACCESS_KEY_ID[-4:]}")

# ── Inject into environment for boto3 ────────────────────────────────────────
os.environ["AWS_ACCESS_KEY_ID"]     = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"]    = AWS_REGION

# ── Now run the main launcher ─────────────────────────────────────────────────
import boto3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--tail-only",     action="store_true")
    parser.add_argument("--no-upload",     action="store_true", help="Skip data upload (already in S3)")
    args = parser.parse_args()

    # Verify credentials work
    try:
        sts = boto3.client("sts", region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        print(f"\nAuthenticated: {identity['Arn']}")
        print(f"Account ID  : {identity['Account']}")
    except Exception as e:
        print(f"\n[AUTH ERROR] {e}")
        print("\nCheck that your Access Key ID and Secret are correct in .env.aws")
        sys.exit(1)

    # Import and run the launcher
    sys.path.insert(0, str(BASE_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location("launcher",
                str(BASE_DIR / "aws_launch_pipeline.py"))
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    clients = launcher.get_clients(AWS_REGION)
    ec2, s3 = clients["ec2"], clients["s3"]
    bkt = launcher.bucket_name(AWS_REGION)

    print(f"\nS3 Bucket: {bkt}")

    if args.download_only:
        print("\n[DOWNLOAD MODE]")
        launcher.download_outputs(s3, bkt, AWS_REGION)
        return

    # Create bucket
    print("\n[Step 1] S3 Bucket")
    launcher.ensure_bucket(s3, bkt, AWS_REGION)

    if args.tail_only:
        print("\n[TAIL MODE] Watching logs ...")
        launcher.tail_log(s3, bkt, AWS_REGION, instance_id="", poll_interval=20)
        return

    # Upload data files (skip if already done)
    if not args.no_upload:
        print("\n[Step 2] Uploading Data Files")
        data_map = [
            (BASE_DIR/"data"/"raw"/"test"/"test_source1.tsv",              "data/raw/test/test_source1.tsv"),
            (BASE_DIR/"data"/"raw"/"test"/"test_source2.tsv",              "data/raw/test/test_source2.tsv"),
            (BASE_DIR/"data"/"raw"/"test"/"test_source3.tsv",              "data/raw/test/test_source3.tsv"),
            (BASE_DIR/"data"/"processed"/"test"/"test_source1.parquet",    "data/processed/test/test_source1.parquet"),
            (BASE_DIR/"data"/"processed"/"test"/"test_source2.parquet",    "data/processed/test/test_source2.parquet"),
            (BASE_DIR/"data"/"processed"/"test"/"test_source3.parquet",    "data/processed/test/test_source3.parquet"),
            (BASE_DIR/"data"/"processed"/"test_corpus_tfidf_matrix.npz",   "cache/test_corpus_tfidf_matrix.npz"),
        ]
        for local, key in data_map:
            if Path(local).exists():
                launcher.upload_if_not_exists(s3, Path(local), bkt, key)
            else:
                print(f"  [SKIP] {local} not found")

    # Upload scripts
    print("\n[Step 3] Uploading Pipeline Script")
    launcher.upload_scripts(s3, bkt, AWS_REGION)

    # Infrastructure
    print("\n[Step 4] AWS Infrastructure")
    ami_id  = launcher.find_latest_dlami(ec2, AWS_REGION)
    key_pem = launcher.ensure_key_pair(ec2, launcher.KEY_NAME, AWS_REGION)
    sg_id   = launcher.ensure_security_group(ec2, launcher.SG_NAME)

    # Launch spot
    print(f"\n[Step 5] Launching {launcher.INSTANCE_TYPE} Spot Instance")
    mode, iid = launcher.launch_spot(ec2, ami_id, sg_id, launcher.KEY_NAME, bkt, AWS_REGION,
                                      access_key=AWS_ACCESS_KEY_ID,
                                      secret_key=AWS_SECRET_ACCESS_KEY)
    launcher._wait_for_instance(ec2, iid)

    ip_info = ec2.describe_instances(InstanceIds=[iid])
    pub_ip  = ip_info["Reservations"][0]["Instances"][0].get("PublicIpAddress","pending")

    print(f"\n{'='*60}")
    print(f"INSTANCE LAUNCHED  ({mode.upper()})")
    print(f"{'='*60}")
    print(f"  Instance ID : {iid}")
    print(f"  Public IP   : {pub_ip}")
    print(f"  Key File    : {key_pem}")
    print(f"  SSH Command : ssh -i {key_pem} ubuntu@{pub_ip}")
    print(f"  S3 Bucket   : s3://{bkt}/")
    print(f"  Live Logs   : s3://{bkt}/logs/run.log (appears after ~3 min)")
    print(f"\n  Estimated cost: ~$0.90 to $1.50 for full pipeline (6–10 hr at $0.15/hr Spot)")
    print(f"  Instance SELF-TERMINATES after pipeline finishes.")
    print(f"\n  Monitoring live log stream (Ctrl+C to stop watching)...")
    print(f"{'='*60}\n")

    # Tail logs
    launcher.tail_log(s3, bkt, AWS_REGION, iid, poll_interval=20, timeout_min=600)

    # Download
    print("\n[Step 6] Downloading Results")
    launcher.download_outputs(s3, bkt, AWS_REGION)
    print("\nDone!")

if __name__ == "__main__":
    main()
