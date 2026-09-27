#!/usr/bin/env python3
"""
aws_launch_pipeline.py
======================
Full AWS automation for the ML Challenge 2026 pipeline.

What this script does:
 1. Validates AWS credentials & region
 2. Creates an S3 bucket and uploads all required data files (parquet + caches)
 3. Finds the latest AWS Deep Learning AMI (Ubuntu 22.04 + PyTorch + CUDA)
 4. Creates a Key Pair and Security Group (SSH only)
 5. Launches a g4dn.xlarge SPOT instance (1× T4 GPU, 16 GB VRAM, ~$0.15/hr)
 6. Uploads a self-contained user-data bootstrap script that:
      - Mounts the 125 GB NVMe SSD at /mnt/nvme
      - Downloads data from S3
      - Installs dependencies
      - Runs generate_submission.py
      - Uploads matching_results.tsv + all cache files BACK to S3
      - AUTOMATICALLY TERMINATES the instance to stop billing
 7. Tails the CloudWatch log stream so you can watch live progress

Usage:
    python aws_launch_pipeline.py --region ap-south-1

    # After it finishes, download outputs:
    python aws_launch_pipeline.py --download-only
"""

import argparse, boto3, json, os, sys, time
from pathlib import Path

# ── Path references ───────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
PROC_DIR  = DATA_DIR / "processed"
RAW_DIR   = DATA_DIR / "raw"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_REGION    = "ap-south-1"   # Mumbai — close to India, cheap spot
BUCKET_PREFIX     = "mlchallenge26"
INSTANCE_TYPE     = "g4dn.xlarge"  # 1× T4 16 GB VRAM, 4 vCPU, 16 GB RAM
MAX_SPOT_PRICE    = "0.35"         # $0.35/hr cap (on-demand is $0.526/hr)
KEY_NAME          = "mlchallenge26-key"
SG_NAME           = "mlchallenge26-sg"
LOG_GROUP         = "/mlchallenge26/pipeline"

def get_clients(region):
    session = boto3.session.Session(region_name=region)
    return {
        "ec2":   session.client("ec2"),
        "s3":    session.client("s3"),
        "iam":   session.client("iam"),
        "logs":  session.client("logs"),
        "ssm":   session.client("ssm"),
    }

def bucket_name(region):
    import hashlib
    acct = boto3.client("sts").get_caller_identity()["Account"]
    h = hashlib.md5(acct.encode()).hexdigest()[:6]
    return f"{BUCKET_PREFIX}-{h}-{region}"

def ensure_bucket(s3, name, region):
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=name)
        else:
            s3.create_bucket(Bucket=name,
                CreateBucketConfiguration={"LocationConstraint": region})
        print(f"  Created bucket: s3://{name}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  Bucket exists : s3://{name}")

def upload_if_not_exists(s3, local_path: Path, bucket: str, key: str):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        size_mb = local_path.stat().st_size / 1e6
        print(f"  [SKIP already] s3://{bucket}/{key}  ({size_mb:.0f} MB)")
        return
    except Exception:
        pass
    size_mb = local_path.stat().st_size / 1e6
    print(f"  Uploading {local_path.name}  ({size_mb:.0f} MB) → s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key,
                   ExtraArgs={"StorageClass": "STANDARD"},
                   Callback=lambda n: None)
    print(f"  ✓ Done")

def find_latest_dlami(ec2, region):
    """Finds the latest AWS Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04, x86)."""
    resp = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name",             "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*22.04*"]},
            {"Name": "architecture",     "Values": ["x86_64"]},
            {"Name": "state",            "Values": ["available"]},
        ]
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Deep Learning AMI found — try us-east-1 or ap-southeast-1")
    img = images[0]
    print(f"  AMI: {img['ImageId']}  ({img['Name'][:70]})")
    return img["ImageId"]

def ensure_key_pair(ec2, key_name: str, region: str):
    key_file = BASE_DIR / f"{key_name}.pem"
    try:
        ec2.describe_key_pairs(KeyNames=[key_name])
        print(f"  Key pair '{key_name}' already exists.")
        if not key_file.exists():
            print(f"  WARNING: {key_file} not found locally — you may not be able to SSH.")
        return str(key_file)
    except ec2.exceptions.ClientError:
        pass
    kp = ec2.create_key_pair(KeyName=key_name)
    key_file.write_text(kp["KeyMaterial"])
    # Set permissions (works on WSL / Git Bash; ignored on Windows PowerShell)
    try:
        os.chmod(str(key_file), 0o400)
    except Exception:
        pass
    print(f"  Created key pair '{key_name}'. Saved PEM to {key_file}")
    return str(key_file)

def ensure_security_group(ec2, sg_name: str):
    vpcs = ec2.describe_vpcs(Filters=[{"Name":"isDefault","Values":["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    existing = ec2.describe_security_groups(
        Filters=[{"Name":"group-name","Values":[sg_name]},
                 {"Name":"vpc-id",    "Values":[vpc_id]}])["SecurityGroups"]
    if existing:
        sg_id = existing[0]["GroupId"]
        print(f"  Security group '{sg_name}' exists: {sg_id}")
        return sg_id

    sg = ec2.create_security_group(
        GroupName=sg_name,
        Description="ML Challenge 26 - SSH only",
        VpcId=vpc_id)
    sg_id = sg["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0",
                          "Description": "SSH from anywhere"}]
        }])
    print(f"  Created security group: {sg_id}")
    return sg_id

def build_userdata(bucket: str, region: str, access_key: str, secret_key: str):
    """
    Self-contained bash script injected as EC2 user-data.
    Credentials are injected as env-vars so aws-cli and boto3 work
    on the instance without needing an IAM instance profile/role.
    """
    return f"""#!/bin/bash
set -euo pipefail
exec > /var/log/mlchallenge.log 2>&1

# ── Inject AWS credentials so aws-cli and boto3 work without an IAM role ──────
export AWS_ACCESS_KEY_ID="{access_key}"
export AWS_SECRET_ACCESS_KEY="{secret_key}"
export AWS_DEFAULT_REGION="{region}"
export AWS_S3_BUCKET="{bucket}"
export AWS_REGION="{region}"

echo "======================================================"
echo "ML Challenge 26 Pipeline - Boot Script"
echo "======================================================"
date

# ── 1. Mount NVMe SSD (125 GB built into g4dn.xlarge) for fast I/O ──────────
NVME_DEV=$(lsblk -d -o NAME,TYPE | grep disk | grep -v xvd | awk '{{print "/dev/"$1}}' | head -1)
echo "NVMe device: $NVME_DEV"
if [ -b "$NVME_DEV" ]; then
    mkfs.ext4 -F "$NVME_DEV"
    mkdir -p /mnt/nvme
    mount "$NVME_DEV" /mnt/nvme
    echo "Mounted $NVME_DEV at /mnt/nvme"
fi

# ── 2. Set up working directories ─────────────────────────────────────────────
WORK=/mnt/nvme/mlchallenge26
mkdir -p "$WORK/data/raw/test" "$WORK/data/processed/test" "$WORK/submissions" "$WORK/output" "$WORK/cache"

# ── 3. Download data from S3 ──────────────────────────────────────────────────
echo "Downloading data from S3 ..."
aws s3 cp s3://{bucket}/data/raw/test/test_source1.tsv  "$WORK/data/raw/test/"  --region {region}
aws s3 cp s3://{bucket}/data/raw/test/test_source2.tsv  "$WORK/data/raw/test/"  --region {region}
aws s3 cp s3://{bucket}/data/raw/test/test_source3.tsv  "$WORK/data/raw/test/"  --region {region}

# Download parquet files if available (faster loading)
aws s3 cp s3://{bucket}/data/processed/test/ "$WORK/data/processed/test/" --recursive --region {region} || true

# Download pre-computed TF-IDF cache if available
aws s3 cp s3://{bucket}/cache/tfidf_vectorizer.joblib     "$WORK/cache/"  --region {region} || true
aws s3 cp s3://{bucket}/cache/test_corpus_tfidf_matrix.npz "$WORK/cache/" --region {region} || true
aws s3 cp s3://{bucket}/cache/test_s1_tfidf_matrix.npz    "$WORK/cache/"  --region {region} || true
aws s3 cp s3://{bucket}/cache/tfidf_candidates_rolling.pkl "$WORK/cache/" --region {region} || true
aws s3 cp s3://{bucket}/cache/progress.json                "$WORK/cache/"  --region {region} || true

# ── 4. Activate Deep Learning conda env ───────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# ── 5. Install extra dependencies ─────────────────────────────────────────────
echo "Installing Python packages ..."
pip install -q sentence-transformers scipy scikit-learn pyarrow tqdm psutil joblib

# ── 6. Download & patch submission script ──────────────────────────────────────
aws s3 cp s3://{bucket}/scripts/generate_submission_aws.py "$WORK/" --region {region}

# ── 7. Run pipeline ────────────────────────────────────────────────────────────
echo "Starting ML pipeline ..."
cd "$WORK"
python generate_submission_aws.py
echo "Pipeline finished at $(date)"

# ── 8. Upload outputs to S3 ───────────────────────────────────────────────────
echo "Uploading results to S3 ..."
aws s3 cp "$WORK/submissions/matching_results.tsv"         s3://{bucket}/output/ --region {region} || true
aws s3 cp "$WORK/cache/tfidf_vectorizer.joblib"            s3://{bucket}/cache/  --region {region} || true
aws s3 cp "$WORK/cache/test_corpus_tfidf_matrix.npz"       s3://{bucket}/cache/  --region {region} || true
aws s3 cp "$WORK/cache/test_s1_tfidf_matrix.npz"           s3://{bucket}/cache/  --region {region} || true
aws s3 cp "$WORK/cache/tfidf_candidates_rolling.pkl"        s3://{bucket}/cache/  --region {region} || true
aws s3 cp "$WORK/cache/progress.json"                      s3://{bucket}/cache/  --region {region} || true
aws s3 cp "/var/log/mlchallenge.log"                       s3://{bucket}/logs/run.log --region {region} || true

echo "All outputs uploaded to s3://{bucket}/"

# ── 9. Self-terminate to stop billing ────────────────────────────────────────
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
echo "Self-terminating instance $INSTANCE_ID ..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region {region}
echo "DONE."
"""

def _pick_subnet(ec2) -> str:
    """Pick the first available subnet in the default VPC."""
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                 {"Name": "state",  "Values": ["available"]}]
    )["Subnets"]
    # Prefer subnets with most available IPs
    subnets.sort(key=lambda s: s["AvailableIpAddressCount"], reverse=True)
    chosen = subnets[0]["SubnetId"]
    print(f"  Auto-selected subnet: {chosen} ({subnets[0]['AvailabilityZone']})")
    return chosen

def launch_instance(ec2, ami_id, sg_id, key_name, bucket, region,
                    access_key: str, secret_key: str):
    """
    Tries to launch a Spot instance first (cheapest).
    On quota / eligibility errors, automatically falls back to On-Demand.
    Always specifies an explicit SubnetId to avoid Free-Tier AMI restrictions.
    """
    import base64
    userdata_b64 = base64.b64encode(
        build_userdata(bucket, region, access_key, secret_key).encode()
    ).decode()

    subnet_id = _pick_subnet(ec2)

    common = dict(
        ImageId    = ami_id,
        InstanceType = INSTANCE_TYPE,
        KeyName    = key_name,
        SubnetId   = subnet_id,          # explicit subnet avoids Free-Tier AMI check
        SecurityGroupIds = [sg_id],
        UserData   = userdata_b64,
        MinCount   = 1,
        MaxCount   = 1,
        BlockDeviceMappings = [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 50, "VolumeType": "gp3",
                    "DeleteOnTermination": True}
        }],
    )

    # ── 1st attempt: Spot ───────────────────────────────────────────────────────
    SPOT_FALLBACK_CODES = {
        "MaxSpotInstanceCountExceeded",
        "InsufficientInstanceCapacity",
        "SpotMaxPriceTooLow",
        "InvalidParameterCombination",   # free-tier restriction on some accounts
        "Unsupported",
    }
    try:
        print(f"  Attempting Spot instance (max ${MAX_SPOT_PRICE}/hr) ...")
        resp = ec2.run_instances(**common,
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {
                    "MaxPrice":         MAX_SPOT_PRICE,
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                }
            }
        )
        iid = resp["Instances"][0]["InstanceId"]
        print(f"  Spot instance launched: {iid}")
        return "spot", iid

    except ec2.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        if code in SPOT_FALLBACK_CODES:
            print(f"  Spot unavailable ({code}: {msg[:80]})")
            print(f"  Falling back to On-Demand ...")
            print(f"  On-Demand g4dn.xlarge = $0.526/hr (~$3.15 for 6-hr run, well within $200 budget)")
        else:
            raise

    # ── 2nd attempt: On-Demand ─────────────────────────────────────────────────
    resp = ec2.run_instances(**common)
    iid  = resp["Instances"][0]["InstanceId"]
    print(f"  On-Demand instance launched: {iid}")
    return "on-demand", iid


def _wait_for_instance(ec2, iid: str):
    """Block until instance is in 'running' state."""
    print("  Waiting for instance to reach 'running' state ...")
    for attempt in range(40):
        time.sleep(10)
        r = ec2.describe_instances(InstanceIds=[iid])
        state = r["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"    [{attempt+1}] State: {state}")
        if state == "running":
            return
        if state in ("terminated", "shutting-down"):
            raise RuntimeError(f"Instance {iid} went to unexpected state: {state}")
    raise TimeoutError(f"Instance {iid} not running after 400s")

# Keep old name as alias for callers
def launch_spot(ec2, ami_id, sg_id, key_name, bucket, region,
                access_key="", secret_key=""):
    return launch_instance(ec2, ami_id, sg_id, key_name, bucket, region,
                           access_key, secret_key)

def upload_scripts(s3, bucket, region):
    """Upload the generate_submission_aws.py pipeline script to S3.
    Always re-uploads (script is tiny, ensures latest version is on S3)."""
    src = BASE_DIR / "src" / "models" / "generate_submission_aws.py"
    if not src.exists():
        src = BASE_DIR / "src" / "models" / "generate_submission.py"
    mb = src.stat().st_size / 1e6
    print(f"  Uploading {src.name}  ({mb:.1f} MB) → s3://{bucket}/scripts/generate_submission_aws.py")
    s3.upload_file(str(src), bucket, "scripts/generate_submission_aws.py")
    print(f"  ✓ Done")

def tail_log(s3, bucket, region, instance_id, poll_interval=30, timeout_min=180):
    """Poll s3://{bucket}/logs/run.log and print new lines until 'DONE' appears."""
    print(f"\nTailing pipeline log from S3 (polling every {poll_interval}s) ...")
    print("Press Ctrl+C to stop tailing (instance will continue running).\n")
    seen_bytes = 0
    deadline   = time.time() + timeout_min * 60
    try:
        while time.time() < deadline:
            time.sleep(poll_interval)
            try:
                obj = s3.get_object(Bucket=bucket, Key="logs/run.log",
                                    Range=f"bytes={seen_bytes}-")
                body = obj["Body"].read().decode("utf-8", errors="replace")
                if body:
                    print(body, end="", flush=True)
                    seen_bytes += len(body.encode("utf-8"))
                    if "DONE." in body or "Self-terminating" in body:
                        print("\n[PIPELINE COMPLETE]")
                        return
            except Exception:
                pass   # log not yet uploaded — keep waiting
    except KeyboardInterrupt:
        print("\nStopped tailing. Instance still running in background.")

def download_outputs(s3, bucket, region):
    out = BASE_DIR / "submissions" / "matching_results_aws.tsv"
    print(f"Downloading matching_results.tsv from S3 ...")
    s3.download_file(bucket, "output/matching_results.tsv", str(out))
    print(f"  Saved to: {out}")
    # Also pull updated caches
    for key, dest in [
        ("cache/tfidf_vectorizer.joblib",     PROC_DIR / "tfidf_vectorizer.joblib"),
        ("cache/test_corpus_tfidf_matrix.npz", PROC_DIR / "test_corpus_tfidf_matrix.npz"),
        ("cache/test_s1_tfidf_matrix.npz",     PROC_DIR / "test_s1_tfidf_matrix.npz"),
        ("cache/tfidf_candidates_rolling.pkl",  PROC_DIR / "tfidf_candidates_rolling.pkl"),
    ]:
        try:
            s3.download_file(bucket, key, str(dest))
            print(f"  Cache: {dest.name}")
        except Exception as e:
            print(f"  (cache {key} not available: {e})")

def main():
    parser = argparse.ArgumentParser(description="AWS ML Challenge 26 pipeline launcher")
    parser.add_argument("--region",        default=DEFAULT_REGION)
    parser.add_argument("--download-only", action="store_true",
                        help="Skip launch, just download outputs from S3")
    parser.add_argument("--no-upload-data", action="store_true",
                        help="Skip uploading data files (assume already in S3)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"ML Challenge 26 AWS Launcher — region: {args.region}")
    print(f"{'='*60}\n")

    clients = get_clients(args.region)
    ec2, s3 = clients["ec2"], clients["s3"]
    bkt = bucket_name(args.region)

    # ── Credentials check ─────────────────────────────────────────────────────
    try:
        identity = boto3.client("sts", region_name=args.region).get_caller_identity()
        print(f"Authenticated as: {identity['Arn']}")
    except Exception as e:
        print(f"\n[ERROR] AWS credentials not configured: {e}")
        print("\nRun:  aws configure")
        print("Then enter your Access Key ID, Secret Access Key, and region.\n")
        sys.exit(1)

    # ── S3 bucket ─────────────────────────────────────────────────────────────
    print("\n[Step 1] S3 Bucket")
    ensure_bucket(s3, bkt, args.region)

    if args.download_only:
        print("\n[Download Mode]")
        download_outputs(s3, bkt, args.region)
        return

    # ── Upload data ───────────────────────────────────────────────────────────
    if not args.no_upload_data:
        print("\n[Step 2] Uploading Data Files to S3")
        uploads = [
            # Raw TSV files
            (RAW_DIR  / "test" / "test_source1.tsv",                 "data/raw/test/test_source1.tsv"),
            (RAW_DIR  / "test" / "test_source2.tsv",                 "data/raw/test/test_source2.tsv"),
            (RAW_DIR  / "test" / "test_source3.tsv",                 "data/raw/test/test_source3.tsv"),
            # Parquet versions (faster loading)
            (PROC_DIR / "test" / "test_source1.parquet",             "data/processed/test/test_source1.parquet"),
            (PROC_DIR / "test" / "test_source2.parquet",             "data/processed/test/test_source2.parquet"),
            (PROC_DIR / "test" / "test_source3.parquet",             "data/processed/test/test_source3.parquet"),
            # Pre-computed TF-IDF caches (HUGE time saver!)
            (PROC_DIR / "test_corpus_tfidf_matrix.npz",              "cache/test_corpus_tfidf_matrix.npz"),
        ]
        for local, key in uploads:
            if Path(local).exists():
                upload_if_not_exists(s3, Path(local), bkt, key)
            else:
                print(f"  [SKIP not found] {local}")

    # ── Upload pipeline script ────────────────────────────────────────────────
    print("\n[Step 3] Uploading Pipeline Script")
    upload_scripts(s3, bkt, args.region)

    # ── AMI, Key Pair, Security Group ─────────────────────────────────────────
    print("\n[Step 4] Infrastructure Setup")
    ami_id = find_latest_dlami(ec2, args.region)
    key_pem = ensure_key_pair(ec2, KEY_NAME, args.region)
    sg_id   = ensure_security_group(ec2, SG_NAME)

    # ── Spot Instance ─────────────────────────────────────────────────────────
    print(f"\n[Step 5] Launching Spot Instance ({INSTANCE_TYPE}, max ${MAX_SPOT_PRICE}/hr)")
    sir, iid = launch_spot(ec2, ami_id, sg_id, KEY_NAME, bkt, args.region)

    ip_info = ec2.describe_instances(InstanceIds=[iid])
    pub_ip = ip_info["Reservations"][0]["Instances"][0].get("PublicIpAddress","pending")
    print(f"\nInstance ID : {iid}")
    print(f"Public IP   : {pub_ip}")
    print(f"SSH access  : ssh -i {key_pem} ubuntu@{pub_ip}")
    print(f"Log file    : s3://{bkt}/logs/run.log  (appears after ~5 min)")
    print(f"\nInstance will SELF-TERMINATE after pipeline completes to stop billing.\n")

    # ── Tail logs ─────────────────────────────────────────────────────────────
    tail_log(s3, bkt, args.region, iid)

    # ── Download results ──────────────────────────────────────────────────────
    print("\n[Step 6] Downloading Results")
    download_outputs(s3, bkt, args.region)
    print(f"\nDone! Cost: ~${0.15 * 6:.2f} (estimated 6-hr run at $0.15/hr Spot)")

if __name__ == "__main__":
    main()
