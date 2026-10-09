import os
import sys
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from settings import (
    R2_ACCOUNT_ID,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME,
)

###############################################################################
# Cloudflare R2 Object Storage Manager
#
# Functional Details:
# - Connects to Cloudflare R2 Object Storage via S3-compatible boto3 API.
# - Manages persistent, cloud-based storage for generated markdown travel plans.
# - Provides functions for:
#   * is_r2_configured: Checks if valid R2 credentials and bucket are present.
#   * get_r2_client: Initializes thread-safe boto3 S3 client with v4 signature.
#   * upload_plan_to_r2: Uploads markdown travel plans to R2 bucket.
#   * list_r2_plans: Retrieves list of previously generated plans from R2.
#   * download_r2_plan: Reads and returns stored plan text from R2.
# - Ensures seamless persistence on stateless/ephemeral cloud deployments (e.g. Streamlit Cloud).
###############################################################################

_cached_client = None


def is_r2_configured() -> bool:
    """Returns True if all required Cloudflare R2 credentials are provided."""
    return bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME)


def get_r2_client():
    """Initializes and returns a boto3 S3 client for Cloudflare R2."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    if not is_r2_configured():
        return None

    try:
        endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        client = boto3.client(
            service_name="s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
            region_name="auto",
        )
        _cached_client = client
        return _cached_client
    except Exception as e:
        print(f"[ERROR] Failed to initialize Cloudflare R2 client: {e}")
        return None


def upload_plan_to_r2(filename: str, content: str, prefix: str = "travel_plans/") -> dict:
    """
    Uploads a travel plan markdown string to Cloudflare R2.
    
    Returns:
        dict: {
            "success": bool,
            "key": str,
            "bucket": str,
            "error": str or None
        }
    """
    if not is_r2_configured():
        return {
            "success": False,
            "key": "",
            "bucket": "",
            "error": "Cloudflare R2 is not configured in environment or Streamlit secrets."
        }

    client = get_r2_client()
    if not client:
        return {
            "success": False,
            "key": "",
            "bucket": "",
            "error": "Could not create Cloudflare R2 client."
        }

    # Ensure clean key path
    clean_prefix = prefix.strip("/")
    key = f"{clean_prefix}/{filename}" if clean_prefix else filename

    try:
        content_bytes = content.encode("utf-8")
        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=key,
            Body=content_bytes,
            ContentType="text/markdown; charset=utf-8",
            Metadata={
                "generated-by": "AI-Travel-MCP-System",
                "filename": filename,
            },
        )
        print(f"[OK] Successfully uploaded {key} to Cloudflare R2 bucket '{R2_BUCKET_NAME}'")
        return {
            "success": True,
            "key": key,
            "bucket": R2_BUCKET_NAME,
            "error": None
        }
    except ClientError as e:
        err_msg = f"Cloudflare R2 ClientError: {e.response.get('Error', {}).get('Message', str(e))}"
        print(f"[ERROR] {err_msg}")
        return {
            "success": False,
            "key": key,
            "bucket": R2_BUCKET_NAME,
            "error": err_msg
        }
    except Exception as e:
        err_msg = f"Upload failed: {str(e)}"
        print(f"[ERROR] {err_msg}")
        return {
            "success": False,
            "key": key,
            "bucket": R2_BUCKET_NAME,
            "error": err_msg
        }


def list_r2_plans(prefix: str = "travel_plans/", max_keys: int = 50) -> list:
    """Lists travel plans stored in Cloudflare R2 bucket."""
    if not is_r2_configured():
        return []

    client = get_r2_client()
    if not client:
        return []

    try:
        response = client.list_objects_v2(
            Bucket=R2_BUCKET_NAME,
            Prefix=prefix,
            MaxKeys=max_keys,
        )
        contents = response.get("Contents", [])
        return [
            {
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S"),
            }
            for obj in contents
        ]
    except Exception as e:
        print(f"[WARN] Failed to list R2 objects: {e}")
        return []


def download_r2_plan(key: str) -> str:
    """Downloads and returns the text content of a travel plan from Cloudflare R2."""
    if not is_r2_configured():
        return ""

    client = get_r2_client()
    if not client:
        return ""

    try:
        response = client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        return response["Body"].read().decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to download {key} from R2: {e}")
        return ""


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n--- Cloudflare R2 Configuration Test ---")
    print(f"Is Configured: {is_r2_configured()}")
    print(f"Bucket Name  : {R2_BUCKET_NAME}")
    print(f"Account ID   : {R2_ACCOUNT_ID[:6]}***" if R2_ACCOUNT_ID else "Account ID   : [MISSING]")

    if is_r2_configured():
        print("\nTesting connection & listing bucket objects...")
        plans = list_r2_plans()
        print(f"Found {len(plans)} objects in '{R2_BUCKET_NAME}/travel_plans/':")
        for p in plans[:5]:
            print(f" - {p['key']} ({p['size']} bytes, modified {p['last_modified']})")

        test_file = "test_connectivity_ping.md"
        test_body = "# R2 Connectivity Test\nCloudflare R2 is working!"
        print(f"\nTesting upload of '{test_file}'...")
        res = upload_plan_to_r2(test_file, test_body, prefix="travel_plans/tests")
        print(f"Upload Result: {res}")
