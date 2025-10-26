from pathlib import Path
from abstract_upload import UploadTemplate


class HDF5Upload(UploadTemplate):
    """HDF5 file uploader for EVE embodiment."""
    
    def __init__(self):
        """
        Initialize HDF5 uploader for EVE embodiment.
        Directory will be prompted from user during file discovery.
        """
        super().__init__("eve")
        self.datatype = ".hdf5"
    
    def verifylocal(self):
        """
        Discover HDF5 files in the local directory.
        Adds all valid HDF5 files to the processing queue.
        Call set_directory() first to select the directory.
        """
        # Check if directory has been set
        if not self.directory_prompted or not self.local_dir:
            raise ValueError("Directory not set. Please call set_directory() first.")
            
        hdf5_files = [
            file for file in self.local_dir.iterdir() 
            if file.suffix == self.datatype and file.is_file()
        ]
        
        for hdf5_file in hdf5_files:
            self.file_paths.append(hdf5_file)
            print(f"Found HDF5 file: {hdf5_file.name}")
        
        print(f"\nDiscovered {len(self.file_paths)} HDF5 files to process.")

    def run(self):
        """
        Process and upload all discovered HDF5 files.
        First collects all metadata from user, then uploads everything at once.
        """
        
        file_metadata_pairs = []
        
        for hdf5_file in self.file_paths:
            metadict, stamped_name = self.prompt_user(hdf5_file)
            file_metadata_pairs.append((hdf5_file, metadict, stamped_name))
        
        for hdf5_file, metadict, stamped_name in file_metadata_pairs:
            self.upload_metadata(stamped_name, metadict)
            self.upload_file(hdf5_file, stamped_name + hdf5_file.suffix)

def main():
    """Main entry point for HDF5 uploader."""
    # Initialize uploader
    uploader = HDF5Upload()
    
    # Set directory first
    uploader.set_directory()
    
    # Verify files
    uploader.verifylocal()

    uploader.run()


if __name__ == "__main__":
    main()