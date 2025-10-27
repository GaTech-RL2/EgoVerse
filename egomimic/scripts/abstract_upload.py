import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from egomimic.utils.aws.aws_sql import TableRow
import boto3
import asyncio

class Uploader():

    def __init__(self, embodiment, datatype, collect_files):
        """Initialize upload template with S3 configuration and metadata settings."""

        self.embodiment = embodiment
        self.datatype = datatype
        self.s3 = boto3.client("s3")
        self.bucket_name = "rldb"
        self.s3_base_prefix = f"raw_v2/{embodiment}/"
        
        self.collect_files = collect_files
        self.local_dir = None
        self.directory_prompted = False
        
        # Metadata configuration - get keys from TableRow to ensure schema consistency
        self.metadata_keys = [
            field for field in TableRow.__dataclass_fields__.keys() 
            if field not in ["episode_hash", "embodiment", "num_frames", "processed_path", "mp4_path", "is_eval", "eval_score", "eval_success"]
        ]
        
        # Auto-fill and batch functionality
        self.previous_inputs = {}
        self.file_paths = []
        self.batch_metadata = None
        self.use_batch_metadata = False
        self.batch_metadata_asked = False

    async def run(self):
        """Main method to run the uploader."""
        print(f"\n🚀 Starting {self.embodiment.upper()} Uploader")
        
        uploads = []
        self.set_directory()
        
        # Collect all files first to show progress
        all_items = list(self.collect_files(self.local_dir))
        
        if not all_items:
            print(f"❌ No {self.datatype} files found in the directory.")
            print("Please check the directory path and file extensions.")
            return
        
        print(f"\n📁 Found {len(all_items)} file groups to process")
        print("=" * 60)
        
        for i, item in enumerate(all_items, 1):
            print(f"\n📝 Processing file group {i}/{len(all_items)}")
            
            # Handle both single values and tuples
            if isinstance(item, tuple):
                file1, file2 = item
                # Find the main file that matches the specified datatype
                if file1.suffix == self.datatype:
                    main_file = file1
                    extra_file = file2
                elif file2.suffix == self.datatype:
                    main_file = file2
                    extra_file = file1
                else:
                    # Neither matches datatype, use first as main
                    main_file = file1
                    extra_file = file2
                
                print(f"   📄 Main file: {main_file.name}")
                print(f"   📄 Extra file: {extra_file.name}")
            else:
                # Single file
                main_file = item
                extra_file = None
                print(f"   📄 File: {main_file.name}")
            
            # Generate timestamp based on main file
            timestamp = self.get_timestamp_name(main_file)
            print(f"   🕒 Timestamp: {timestamp}")
            
            # Collect metadata for the main file only
            metajsonpath = self.collect_metadata(main_file)
            
            # Upload metadata with timestamp
            uploads.append(asyncio.create_task(
                self.upload_file(metajsonpath, new_name=f"{timestamp}_metadata.json")
            ))
            
            # Upload main file with timestamp and suffix
            uploads.append(asyncio.create_task(
                self.upload_file(main_file, new_name=f"{timestamp}{main_file.suffix}")
            ))
            
            # Upload extra file with same timestamp if it exists
            if extra_file:
                uploads.append(asyncio.create_task(
                    self.upload_file(extra_file, new_name=f"{timestamp}{extra_file.suffix}")
                ))

        print(f"\n🔄 Starting uploads for {len(uploads)} files...")
        print("=" * 60)
        
        await asyncio.gather(*uploads)
        
        print(f"\n✅ Upload completed successfully!")
        print(f"   📊 Processed {len(all_items)} file groups")
        print(f"   ☁️  Uploaded {len(uploads)} files to S3")
        print(f"   🎯 Destination: s3://{self.bucket_name}/{self.s3_base_prefix}")
        print("=" * 60)

    def set_directory(self):
        """
        Prompt user to select the directory containing files to upload.
        Call this method before verifylocal() and run().
        """
        print("\n" + "=" * 60)
        print("📂 DATA DIRECTORY SELECTION")
        print("=" * 60)
        
        while True:
            directory = input("🔍 Enter the path to the directory containing files to upload: ").strip()
            
            if not directory:
                print("❌ Error: Directory path cannot be empty. Please enter a valid path.")
                continue
            
            dir_path = Path(directory).expanduser().resolve()
            
            if not dir_path.exists():
                print(f"❌ Error: Directory '{dir_path}' does not exist. Please enter a valid path.")
                continue
            
            if not dir_path.is_dir():
                print(f"❌ Error: '{dir_path}' is not a directory. Please enter a directory path.")
                continue
            
            print(f"✅ Selected directory: {dir_path}")
            self.local_dir = dir_path
            self.directory_prompted = True
            break

    def collect_metadata(self, file_path: Path):
        """
        Collect metadata for a specific file using CLI input with auto-fill or batch metadata.
        Creates a temporary JSON file with the metadata that can be uploaded using upload_file().
        
        Args:
            file_path: Path to the file being processed
            
        Returns:
            Tuple of (temp_json_file_path, file_timestamp)
        """
        
        # Prompt for directory on first call if not provided during initialization
        if not self.directory_prompted and self.local_dir is None:
            self.set_directory()
        
        # Handle batch metadata prompt on first file only
        if not self.batch_metadata_asked and len(self.file_paths) > 1:
            self.batch_metadata_asked = True
            print(f"\n📊 Found {len(self.file_paths)} files to upload.")
            
            # Create dynamic prompt based on TableRow fields
            field_names = ", ".join(self.metadata_keys)
            batch_prompt = (
                f"🤔 Do all files share the same metadata ({field_names})? (y/N): "
            )
            response = input(batch_prompt)
            
            if response.lower() == 'y':
                print("\n" + "=" * 60)
                print("📋 BATCH METADATA COLLECTION")
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
                
                print(f"\n✅ Batch metadata collected! "
                      f"This will be applied to all {len(self.file_paths)} files.")
                self.use_batch_metadata = True
            else:
                print("\n📝 Will collect metadata for each file individually.")
                self.use_batch_metadata = False
        
        # Collect or use metadata based on selected mode
        if self.use_batch_metadata and self.batch_metadata:
            # Use previously collected batch metadata
            submitted_metadata = self.batch_metadata.copy()
            print(f"   📋 Using batch metadata for: {file_path.name}")
        else:
            # Collect individual metadata for this file
            submitted_metadata = {
                "embodiment": self.embodiment,
                "episode_hash": "",  # Will be set below
            }

            print(f"\n📝 METADATA COLLECTION")
            print(f"File: {file_path.name}")
            print("-" * 50)
            
            if self.previous_inputs:
                print("💡 (Press Enter to keep previous value, or type new value to change)")

            for key in self.metadata_keys:
                previous_value = self.previous_inputs.get(key, "")
                
                value = self._collect_metadata_value(key, previous_value)
                submitted_metadata[key] = value
                self.previous_inputs[key] = value  # Save for auto-fill

        # Add file-specific timestamp using the get_timestamp_name function
        timestamp_ms = self.get_timestamp_name(file_path)
            
        submitted_metadata["episode_hash"] = timestamp_ms  # Use int for episode_hash
        submitted_metadata["embodiment"] = self.embodiment

        # Create temporary JSON file for metadata
        metadata_tempfile = tempfile.NamedTemporaryFile(
            delete=False, mode='w', suffix='.json'
        )
        json.dump(submitted_metadata, metadata_tempfile, indent=2)
        metadata_tempfile.close()
        
        return Path(metadata_tempfile.name)

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

    def get_timestamp_name(self, file_path):
        """
        Generate a timestamp for file identification.
        
        Returns:
            An integer representing the file timestamp in milliseconds.
        """
        stats = os.stat(file_path)

        # Use st_birthtime where available (macOS/BSD), else fall back to st_ctime
        if hasattr(stats, "st_birthtime"):
            timestamp_ms = int(stats.st_birthtime * 1000)
        else:
            # Windows: st_ctime gives creation time
            # Linux: st_ctime gives last metadata change time
            timestamp_ms = int(stats.st_ctime * 1000)

        return timestamp_ms

    async def upload_file(self, file_path, new_name=None):
        """
        Upload a file to S3.

        Args:
            file_path: Path to the local file to upload
            new_name: Optional new name for the file in S3 storage
        """
        s3 = boto3.client("s3")
        s3_key = f"{self.s3_base_prefix}{new_name or Path(file_path).name}"

        print(f"   ☁️  Uploading {Path(file_path).name} → s3://{self.bucket_name}/{s3_key}")
        s3.upload_file(str(file_path), self.bucket_name, s3_key)
        print(f"   ✅ Completed: {Path(file_path).name}")