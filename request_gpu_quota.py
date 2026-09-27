"""
request_gpu_quota.py
=====================
Automatically requests a service quota increase for G-family GPU instances
so that g4dn.xlarge becomes launchable on a new AWS account.

Also requests the Spot quota so both paths are unlocked.
"""
import boto3, os, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("C:/Users/Pavan/Downloads/ML_Challenge_26/.env.aws"))
REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

sq  = boto3.client("service-quotas", region_name=REGION)
iam = boto3.client("iam",            region_name=REGION)

# ── Step 0: Attach ServiceQuotasFullAccess so we can read/write quotas ────────
print("Attaching ServiceQuotasFullAccess to mlchallenge26-runner ...")
try:
    iam.attach_user_policy(
        UserName="mlchallenge26-runner",
        PolicyArn="arn:aws:iam::aws:policy/ServiceQuotasFullAccess"
    )
    print("  Attached OK. Waiting 5s for IAM to propagate ...")
    time.sleep(5)
except Exception as e:
    code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
    if "AccessDenied" in str(e):
        print("  IAM user lacks iam:AttachUserPolicy permission.")
        print("  Do this MANUALLY in the AWS Console (takes 2 minutes):")
        print()
        print("  1. Go to: https://console.aws.amazon.com/iam/home#/users/mlchallenge26-runner")
        print("  2. Click 'Add permissions' -> 'Attach policies directly'")
        print("  3. Search for and check: ServiceQuotasFullAccess")
        print("  4. Click 'Next' -> 'Add permissions'")
        print("  5. Re-run: python request_gpu_quota.py")
        import sys; sys.exit(1)
    print(f"  (Already attached or error: {e})")

# ── g4dn.xlarge needs 4 vCPUs from the "G and VT" quota ─────────────────────
QUOTAS = [
    {
        "QuotaCode":  "L-DB2E81BA",       # Running On-Demand G and VT instances
        "QuotaName":  "Running On-Demand G and VT instances",
        "DesiredValue": 4,                # 4 vCPUs = 1× g4dn.xlarge
    },
    {
        "QuotaCode":  "L-3819A6DF",       # All G and VT Spot Instance Requests
        "QuotaName":  "All G and VT Spot Instance Requests",
        "DesiredValue": 4,
    },
]

print(f"Region: {REGION}")
print(f"Requesting GPU quota increases...\n")

for q in QUOTAS:
    print(f"Quota: {q['QuotaName']}")

    # Check current value first
    try:
        current = sq.get_aws_default_service_quota(
            ServiceCode="ec2", QuotaCode=q["QuotaCode"])
        current_val = current["Quota"]["Value"]
        print(f"  Current default: {current_val} vCPUs")
    except Exception:
        current_val = 0
        print(f"  Current default: unknown")

    try:
        applied = sq.get_service_quota(
            ServiceCode="ec2", QuotaCode=q["QuotaCode"])
        applied_val = applied["Quota"]["Value"]
        print(f"  Current applied: {applied_val} vCPUs")
        if applied_val >= q["DesiredValue"]:
            print(f"  Already sufficient — skipping request.\n")
            continue
    except sq.exceptions.NoSuchResourceException:
        applied_val = 0

    # Submit the increase request
    try:
        resp = sq.request_service_quota_increase(
            ServiceCode="ec2",
            QuotaCode=q["QuotaCode"],
            DesiredValue=q["DesiredValue"]
        )
        case_id = resp["RequestedQuota"].get("CaseId", "N/A")
        status  = resp["RequestedQuota"].get("Status", "PENDING")
        print(f"  Increase requested: {q['DesiredValue']} vCPUs")
        print(f"  Status: {status}   Case ID: {case_id}")
    except sq.exceptions.ResourceAlreadyExistsException:
        print(f"  A pending request already exists for this quota.")
    except Exception as e:
        print(f"  Request failed: {e}")
    print()

print("=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("""
Quota increases for GPU instances on new accounts typically
take 2–24 hours to be approved automatically.

You can check the status at:
  https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas

While waiting, you can:
  1. Run the Kaggle notebook (kaggle_generate_submission.ipynb)
     which uses Kaggle's free T4 GPU — no quotas needed.
  2. Once approved, re-run:
     python aws_configure_and_launch.py --no-upload
""")
