import json
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from egomimic.utils.aws.aws_sql import TableRow
import boto3


class UploadTemplate(ABC):

    def __init__(self, embodiment):
        """Initialize upload template with S3 configuration and metadata settings."""
        # AWS S3 configuration
        self.s3 = boto3.client("s3")
        self.bucket_name = "rldb"
        self.s3_base_prefix = f"raw_v2/{embodiment}/"

        # File system configuration - will be prompted from user
        self.embodiment = embodiment
        self.directory_prompted = False
        self.local_dir = None
        
        # Metadata configuration - get keys from TableRow to ensure schema consistency
        # Get metadata keys from TableRow fields (excluding auto-populated ones)
        self.metadata_keys = [
            field for field in TableRow.__dataclass_fields__.keys() 
            if field not in ["episode_hash", "embodiment", "num_frames", "processed_path", "mp4_path", "is_eval", "eval_score", "eval_success"]
        ]
        
        # Auto-fill functionality
        self.previous_inputs = {}
        
        # File processing storage
        self.file_paths = []
        
        # Batch metadata functionality
        self.batch_metadata = None
        self.use_batch_metadata = False
        self.batch_metadata_asked = False

    def set_directory(self):
        """
        Prompt user to select the directory containing files to upload.
        Call this method before verifylocal() and run().
        """
        print("\n" + "=" * 60)
        print("DATA DIRECTORY SELECTION")
        print("=" * 60)
        
        while True:
            directory = input("Enter the path to the directory containing files to upload: ").strip()
            
            if not directory:
                print("Error: Directory path cannot be empty. Please enter a valid path.")
                continue
            
            dir_path = Path(directory).expanduser().resolve()
            
            if not dir_path.exists():
                print(f"Error: Directory '{dir_path}' does not exist. Please enter a valid path.")
                continue
            
            if not dir_path.is_dir():
                print(f"Error: '{dir_path}' is not a directory. Please enter a directory path.")
                continue
            
            print(f"Selected directory: {dir_path}")
            self.local_dir = dir_path
            self.directory_prompted = True
            break

    @abstractmethod
    def verifylocal(self):
        pass
    
    @abstractmethod
    async def run(self):
        pass

    def prompt_user(self, file_path: Path):
        """
        Collect metadata for a specific file using CLI input with auto-fill or batch metadata.
        On first call, prompts for directory if not provided and asks if all files share metadata.
        
        Args:
            file_path: Path to the file being processed
            
        Returns:
            Tuple of (json_metadata, file_timestamp)
        """
        
        # Prompt for directory on first call if not provided during initialization
        if not self.directory_prompted and self.local_dir is None:
            self.set_directory()
        
        # Handle batch metadata prompt on first file only
        if not self.batch_metadata_asked and len(self.file_paths) > 1:
            self.batch_metadata_asked = True
            print(f"\nFound {len(self.file_paths)} files to upload.")
            
            # Create dynamic prompt based on TableRow fields
            field_names = ", ".join(self.metadata_keys)
            batch_prompt = (
                f"Do all files share the same metadata ({field_names})? (y/N): "
            )
            response = input(batch_prompt)
            
            if response.lower() == 'y':
                print("\n" + "=" * 60)
                print("BATCH METADATA COLLECTION")
                print("Enter metadata that will be applied to ALL files:")
                print("=" * 60)
                
                # Initialize batch metadata from TableRow structure
                self.batch_metadata = {
                    "embodiment": self.embodiment,
                    "episode_hash": "",  # Will be set per file
                }
                
                for key in self.metadata_keys:
                    value = self._collect_metadata_value(key)
                    self.batch_metadata[key] = value
                
                print(f"\nBatch metadata collected! "
                      f"This will be applied to all {len(self.file_paths)} files.")
                self.use_batch_metadata = True
            else:
                print("\nWill collect metadata for each file individually.")
                self.use_batch_metadata = False
        
        # Collect or use metadata based on selected mode
        if self.use_batch_metadata and self.batch_metadata:
            # Use previously collected batch metadata
            submitted_metadata = self.batch_metadata.copy()
            print(f"\nProcessing file: {file_path.name} (using batch metadata)")
        else:
            # Collect individual metadata for this file
            submitted_metadata = {
                "embodiment": self.embodiment,
                "episode_hash": "",  # Will be set below
            }

            print(f"\nFile: {file_path.name}")
            print("-" * 50)
            
            if self.previous_inputs:
                print("(Press Enter to keep previous value, or type new value to change)")

            for key in self.metadata_keys:
                previous_value = self.previous_inputs.get(key, "")
                
                value = self._collect_metadata_value(key, previous_value)
                submitted_metadata[key] = value
                self.previous_inputs[key] = value  # Save for auto-fill

        # Add file-specific timestamp
        stats = os.stat(file_path)

        # Use st_birthtime where available (macOS/BSD), else fall back to st_ctime
        if hasattr(stats, "st_birthtime"):
            timestamp_ms = int(stats.st_birthtime * 1000)
        else:
            # Windows: st_ctime gives creation time
            # Linux: st_ctime gives last metadata change time
            timestamp_ms = int(stats.st_ctime * 1000)

        file_timestamp = str(timestamp_ms)
        submitted_metadata["episode_hash"] = timestamp_ms  # Use int for episode_hash
        submitted_metadata["embodiment"] = self.embodiment

        json_metadata = json.dumps(submitted_metadata)
        
        return json_metadata, file_timestamp

    def _collect_metadata_value(self, key, previous_value=""):
        """
        Collect a metadata value, handling objects as a list.
        
        Args:
            key: The metadata key
            previous_value: Previously entered value for auto-fill
            
        Returns:
            The collected value (string or list for objects)
        """
        while True:
            if key == "objects":
                if previous_value:
                    # Convert previous list back to comma-separated string for display
                    display_value = ", ".join(previous_value) if isinstance(previous_value, list) else previous_value
                    prompt = f"Enter {key} (comma-separated list) [{display_value}]: "
                else:
                    prompt = f"Enter {key} (comma-separated list): "
            else:
                if previous_value:
                    prompt = f"Enter {key} [{previous_value}]: "
                else:
                    prompt = f"Enter {key}: "
            
            value = input(prompt).strip()
            
            # Use previous value if empty input and previous value exists
            if not value and previous_value:
                return previous_value
            
            if value:
                # Special handling for objects - return as list for JSON
                if key == "objects":
                    # Split by comma and clean up each item to create a list
                    objects_list = [item.strip() for item in value.split(",") if item.strip()]
                    if not objects_list:
                        print("Error: objects cannot be empty. Please enter at least one object.")
                        continue
                    return objects_list
                else:
                    return value
            else:
                print(f"Error: {key} cannot be empty. Please enter a value.")

    def upload_metadata(self, file_timestamp, metadict):
        """
        Upload metadata file to S3.
        
        Args:
            file_timestamp: Timestamp string for the file
            metadict: JSON string containing metadata
        """
        s3 = boto3.client("s3")
        file_metadata = json.loads(metadict)
        
        # Generate S3 key for metadata
        metadata_key = f"{self.s3_base_prefix}{file_timestamp}_metadata.json"
        
        # Create temporary file for upload
        metadata_tempfile = tempfile.NamedTemporaryFile(
            delete=False, mode='w', suffix='.json'
        )
        json.dump(file_metadata, metadata_tempfile, indent=2)
        metadata_tempfile.close()

        try:
            print(f"Uploading metadata to s3://{self.bucket_name}/{metadata_key}")
            s3.upload_file(metadata_tempfile.name, self.bucket_name, metadata_key)
        finally:
            # Clean up temporary file
            os.remove(metadata_tempfile.name)

    def upload_file(self, local_file, stamped_name):
        """
        Upload a file to S3.

        Args:
            local_file: Path to the local file to upload
            stamped_name: Timestamped filename for S3 storage
        """
        s3 = boto3.client("s3")
        s3_key = f"{self.s3_base_prefix}{stamped_name}"
        
        print(f"Uploading {Path(local_file).name} to s3://{self.bucket_name}/{s3_key}")
        s3.upload_file(str(local_file), self.bucket_name, s3_key)