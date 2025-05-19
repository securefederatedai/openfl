import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import fnmatch
from pathlib import Path


class S3Helper:
    """Helper class for working with S3 or MinIO buckets."""
    
    def __init__(self, endpoint_url='http://localhost:9000', 
                 access_key=None, secret_key=None, region='us-east-1'):
        """
        Initialize S3Helper with connection details.
        
        Args:
            endpoint_url: The S3 endpoint URL (default: http://localhost:9000 for MinIO)
            access_key: The access key (if None, uses MINIO_ROOT_USER env variable)
            secret_key: The secret key (if None, uses MINIO_ROOT_PASSWORD env variable)
            region: The region name (default: us-east-1, required by boto3 but not used by MinIO)
        """
        self.endpoint_url = endpoint_url
        self.access_key = access_key or os.environ.get('MINIO_ROOT_USER', 'minioadmin')
        self.secret_key = secret_key or os.environ.get('MINIO_ROOT_PASSWORD', 'minioadmin')
        self.region = region
        
        # Initialize S3 client
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name=self.region
        )
        
        # Initialize S3 resource (for higher-level operations)
        self.resource = boto3.resource(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name=self.region
        )
    
    def create_bucket(self, bucket_name):
        """
        Create a new bucket if it doesn't exist.

        Args:
            bucket_name: Name of the bucket to create
            
        Returns:
            bool: True if bucket was created or already exists, False on error
        """
        try:
            # Check if bucket already exists
            self.client.head_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} already exists.")
            return True
        except ClientError as e:
            # If bucket doesn't exist, create it
            if e.response['Error']['Code'] == '404':
                try:
                    self.client.create_bucket(Bucket=bucket_name)
                    print(f"Bucket {bucket_name} created successfully.")
                    return True
                except ClientError as create_error:
                    print(f"Error creating bucket: {create_error}")
                    return False
            else:
                print(f"Error checking bucket: {e}")
                return False
    
    def delete_bucket(self, bucket_name, force=False):
        """
        Delete a bucket.
        
        Args:
            bucket_name: Name of the bucket to delete
            force: If True, delete all objects in the bucket before deletion
            
        Returns:
            bool: True if bucket was deleted, False on error
        """
        try:
            if force:
                # Delete all objects in the bucket first
                self.delete_all_objects(bucket_name)
                
            # Delete the bucket
            self.client.delete_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} deleted successfully.")
            return True
        except ClientError as e:
            print(f"Error deleting bucket {bucket_name}: {e}")
            return False
    
    def list_buckets(self):
        """
        List all buckets.
        
        Returns:
            list: List of bucket names
        """
        try:
            response = self.client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            print(f"Found {len(buckets)} buckets: {', '.join(buckets)}")
            return buckets
        except ClientError as e:
            print(f"Error listing buckets: {e}")
            return []
    
    def upload_file(self, file_path, bucket_name, object_name=None):
        """
        Upload a file to a bucket.
        
        Args:
            file_path: Path to the file to upload
            bucket_name: Name of the bucket
            object_name: S3 object name (if None, uses file_path basename)
            
        Returns:
            bool: True if file was uploaded, False on error
        """
        # If object_name was not specified, use file_path basename
        if object_name is None:
            object_name = Path(file_path).name
        
        try:
            self.client.upload_file(file_path, bucket_name, object_name)
            print(f"File {file_path} uploaded to {bucket_name}/{object_name}")
            return True
        except ClientError as e:
            print(f"Error uploading file {file_path}: {e}")
            return False
    
    def upload_directory(self, dir_path, bucket_name, prefix=""):
        """
        Upload all files from a directory to a bucket.
        
        Args:
            dir_path: Path to the directory to upload
            bucket_name: Name of the bucket
            prefix: Prefix to add to object names
            
        Returns:
            int: Number of files uploaded
        """
        dir_path = Path(dir_path)
        count = 0
        
        if not dir_path.is_dir():
            print(f"Error: {dir_path} is not a directory")
            return count
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path from dir_path
                rel_path = file_path.relative_to(dir_path)
                # Create object name with prefix
                if prefix:
                    object_name = f"{prefix}/{rel_path}"
                else:
                    object_name = str(rel_path)
                
                if self.upload_file(str(file_path), bucket_name, object_name):
                    count += 1
        
        print(f"Uploaded {count} files to {bucket_name} from {dir_path}")
        return count
    
    def download_file(self, bucket_name, object_name, file_path=None):
        """
        Download a file from a bucket.
        
        Args:
            bucket_name: Name of the bucket
            object_name: S3 object name
            file_path: Local path to save the file (if None, uses object_name basename)
            
        Returns:
            bool: True if file was downloaded, False on error
        """
        # If file_path was not specified, use object_name basename
        if file_path is None:
            file_path = Path(object_name).name
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(Path(file_path).parent, exist_ok=True)
            
            self.client.download_file(bucket_name, object_name, file_path)
            print(f"Downloaded {bucket_name}/{object_name} to {file_path}")
            return True
        except ClientError as e:
            print(f"Error downloading {bucket_name}/{object_name}: {e}")
            return False
    
    def download_directory(self, bucket_name, prefix, local_dir=None):
        """
        Download all files with a prefix from a bucket.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Prefix of objects to download
            local_dir: Local directory to save files (if None, uses current dir)
            
        Returns:
            int: Number of files downloaded
        """
        if local_dir is None:
            local_dir = '.'
        
        local_dir = Path(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        
        count = 0
        try:
            # List all objects with the prefix
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    object_name = obj['Key']
                    
                    # Calculate relative path from prefix
                    if prefix and object_name.startswith(prefix):
                        rel_path = object_name[len(prefix):]
                        if rel_path.startswith('/'):
                            rel_path = rel_path[1:]
                    else:
                        rel_path = object_name
                    
                    # Create local file path
                    file_path = local_dir / rel_path
                    
                    if self.download_file(bucket_name, object_name, str(file_path)):
                        count += 1
            
            print(f"Downloaded {count} files from {bucket_name}/{prefix} to {local_dir}")
            return count
        except ClientError as e:
            print(f"Error downloading from {bucket_name}/{prefix}: {e}")
            return count
    
    def list_objects(self, bucket_name, prefix="", recursive=True, max_items=None):
        """
        List objects in a bucket with an optional prefix.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Prefix filter for objects
            recursive: If False, emulates directory listing with delimiters
            max_items: Maximum number of items to return
            
        Returns:
            list: List of object keys
        """
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            
            # Set up pagination parameters
            pagination_config = {}
            if max_items:
                pagination_config['MaxItems'] = max_items
                
            # Set up operation parameters
            operation_params = {
                'Bucket': bucket_name,
                'Prefix': prefix
            }
            
            # If not recursive, use delimiter to emulate directory listing
            if not recursive:
                operation_params['Delimiter'] = '/'
                
            # Get pages of objects
            pages = paginator.paginate(
                **operation_params, 
                PaginationConfig=pagination_config
            )
            
            objects = []
            
            for page in pages:
                # Add objects
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(obj['Key'])
                
                # Add common prefixes (folders) if not recursive
                if not recursive and 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        objects.append(prefix['Prefix'])
            
            print(f"Found {len(objects)} objects in {bucket_name}/{prefix}")
            for obj in objects:
                print(f"- {obj}")
                
            return objects
        except ClientError as e:
            print(f"Error listing objects in {bucket_name}/{prefix}: {e}")
            return []
    
    def delete_object(self, bucket_name, object_name):
        """
        Delete an object from a bucket.
        
        Args:
            bucket_name: Name of the bucket
            object_name: S3 object name to delete
            
        Returns:
            bool: True if object was deleted, False on error
        """
        try:
            self.client.delete_object(
                Bucket=bucket_name,
                Key=object_name
            )
            print(f"Deleted {bucket_name}/{object_name}")
            return True
        except ClientError as e:
            print(f"Error deleting {bucket_name}/{object_name}: {e}")
            return False
    
    def delete_objects(self, bucket_name, object_names):
        """
        Delete multiple objects from a bucket.
        
        Args:
            bucket_name: Name of the bucket
            object_names: List of object names to delete
            
        Returns:
            int: Number of objects deleted
        """
        if not object_names:
            return 0
            
        try:
            # Create delete request
            objects = [{'Key': obj} for obj in object_names]
            response = self.client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': objects}
            )
            
            deleted = len(response.get('Deleted', []))
            errors = len(response.get('Errors', []))
            
            print(f"Deleted {deleted} objects from {bucket_name}")
            if errors > 0:
                print(f"Failed to delete {errors} objects")
                
            return deleted
        except ClientError as e:
            print(f"Error deleting objects from {bucket_name}: {e}")
            return 0
    
    def delete_prefix(self, bucket_name, prefix):
        """
        Delete all objects with a specific prefix (like a folder).
        
        Args:
            bucket_name: Name of the bucket
            prefix: Prefix of objects to delete
            
        Returns:
            int: Number of objects deleted
        """
        try:
            # List all objects with the prefix
            objects = self.list_objects(bucket_name, prefix)
            
            # Delete the objects in batches
            count = 0
            batch_size = 1000  # S3 limits delete_objects to 1000 at a time
            
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                count += self.delete_objects(bucket_name, batch)
            
            print(f"Deleted {count} objects from {bucket_name}/{prefix}")
            return count
        except ClientError as e:
            print(f"Error deleting prefix {bucket_name}/{prefix}: {e}")
            return 0
    
    def delete_all_objects(self, bucket_name):
        """
        Delete all objects in a bucket.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            int: Number of objects deleted
        """
        return self.delete_prefix(bucket_name, "")
    
    def split_directory_to_buckets(self, source_path, bucket_name, folder_names, split_folders=None):
        """
        Split folders from a directory into separate folders in a bucket.
        
        Args:
            source_path: Path to the directory containing folders to split
            bucket_name: Name of the bucket to upload to
            folder_names: List of folder names to upload
            split_folders: Dictionary mapping folders to destination prefixes,
                           if None, splits into equal groups
            
        Returns:
            dict: Mapping of destination prefixes to lists of folders uploaded
        """
        source_path = Path(source_path)
        if not source_path.is_dir():
            print(f"Error: {source_path} is not a directory")
            return {}
            
        # Ensure bucket exists
        self.create_bucket(bucket_name)
        
        # Get folders in source directory that match requested folder names
        folders = []
        for folder_name in folder_names:
            folder_path = source_path / folder_name
            if folder_path.is_dir():
                folders.append(folder_name)
            else:
                print(f"Warning: {folder_path} is not a directory, skipping")
        
        # If split_folders is None, create equal groups
        if split_folders is None:
            half = len(folders) // 2
            split_folders = {
                "1": folders[:half],
                "2": folders[half:]
            }
            
        result = {}
        
        # Upload each group of folders to the specified prefix
        for prefix, group_folders in split_folders.items():
            result[prefix] = []
            
            for folder in group_folders:
                if folder in folders:
                    folder_path = source_path / folder
                    # Upload the folder with the prefix
                    upload_prefix = f"{prefix}/{folder}"
                    count = self.upload_directory(folder_path, bucket_name, upload_prefix)
                    if count > 0:
                        result[prefix].append(folder)
                        print(f"Uploaded {folder} to {bucket_name}/{upload_prefix}")
        
        return result

    def copy_object(self, source_bucket, source_key, dest_bucket, dest_key=None):
        """
        Copy an object within or between buckets.
        
        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key (if None, uses source_key)
            
        Returns:
            bool: True if object was copied, False on error
        """
        if dest_key is None:
            dest_key = source_key
            
        try:
            copy_source = {
                'Bucket': source_bucket,
                'Key': source_key
            }
            
            self.client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            
            print(f"Copied {source_bucket}/{source_key} to {dest_bucket}/{dest_key}")
            return True
        except ClientError as e:
            print(f"Error copying {source_bucket}/{source_key}: {e}")
            return False

    def search_objects(self, bucket_name, pattern, prefix=""):
        """
        Search for objects in a bucket using a glob pattern.
        
        Args:
            bucket_name: Name of the bucket
            pattern: Glob pattern to match object keys against
            prefix: Optional prefix to limit search scope
            
        Returns:
            list: List of matching object keys
        """
        objects = self.list_objects(bucket_name, prefix)
        matches = [obj for obj in objects if fnmatch.fnmatch(obj, pattern)]
        
        print(f"Found {len(matches)} objects matching '{pattern}' in {bucket_name}/{prefix}")
        for obj in matches:
            print(f"- {obj}")
            
        return matches


if __name__ == "__main__":
    # Example usage
    s3 = S3Helper()
    
    # List all buckets
    s3.list_buckets()
    
    # Create a bucket
    s3.create_bucket('mybucket-hist')
    
    # List objects in a bucket
    s3.list_objects('mybucket-hist', prefix='')
    
    # Upload histology data with split directories
    source_path = '/home/azureuser/openfl/torch_s3/Kather_texture_2016_image_tiles_5000'
    
    # Get folder names from source path
    if Path(source_path).is_dir():
        folder_names = [f for f in os.listdir(source_path) 
                        if Path(source_path, f).is_dir()]
        
        # Split folders into equal groups and upload to bucket
        result = s3.split_directory_to_buckets(
            source_path, 
            'mybucket-hist', 
            folder_names
        )
        
        print("Upload result:")
        for prefix, folders in result.items():
            print(f"- {prefix}: {folders}")
