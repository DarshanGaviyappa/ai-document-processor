"""
S3 Storage Manager for organizing and uploading files
"""
import boto3
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


class S3Manager:
    """Manage file uploads to S3 with organized structure"""
    
    def __init__(self):
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
    
    def generate_s3_key(self, file_type: str, source: str, filename: str) -> str:
        """
        Generate organized S3 key path
        
        Structure: {file_type}/{source}/{year-month}/{filename}
        Example: markdown/processed/2025-12/document.md
        """
        date_prefix = datetime.now().strftime("%Y-%m")
        s3_key = f"{file_type}/{source}/{date_prefix}/{filename}"
        return s3_key
    
    def upload_file(self, local_path: str, s3_key: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Upload file to S3 with metadata
        
        Args:
            local_path: Path to local file
            s3_key: S3 object key (path)
            metadata: Optional metadata dict
            
        Returns:
            Dictionary with upload results
        """
        result = {
            "success": False,
            "s3_key": s3_key,
            "s3_url": "",
            "local_path": local_path,
            "error": None
        }
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Upload file
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            # Generate URLs
            result["s3_url"] = f"s3://{self.bucket_name}/{s3_key}"
            result["https_url"] = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in S3 bucket with given prefix"""
        files = []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
        except Exception as e:
            print(f"Error listing files: {e}")
        
        return files


if __name__ == "__main__":
    # Test S3 Manager
    print("=" * 60)
    print("Testing S3 Manager...")
    print("=" * 60)
    
    s3 = S3Manager()
    
    # Find a markdown file to upload
    data_path = Path("data")
    md_files = list(data_path.glob("*.md"))
    
    if md_files:
        md_file = str(md_files[0])
        filename = Path(md_file).name
        
        s3_key = s3.generate_s3_key("markdown", "processed", filename)
        print(f"\nUploading: {md_file}")
        print(f"S3 Key: {s3_key}")
        
        result = s3.upload_file(
            md_file, 
            s3_key,
            metadata={
                'source': 'markitdown',
                'processed_date': datetime.now().isoformat()
            }
        )
        
        if result['success']:
            print(f"✓ Upload successful!")
            print(f"✓ S3 URL: {result['s3_url']}")
            print(f"✓ HTTPS URL: {result['https_url']}")
        else:
            print(f"✗ Upload failed: {result['error']}")
    else:
        print("No markdown files found to test")
    
    # List all files
    print("\n" + "=" * 60)
    print("Listing files in S3 bucket...")
    print("=" * 60)
    files = s3.list_files()
    if files:
        for f in files:
            print(f"- {f['key']} ({f['size']} bytes, {f['last_modified']})")
    else:
        print("No files in bucket yet")