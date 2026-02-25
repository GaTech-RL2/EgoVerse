import asyncio

from abstract_upload import Uploader


def eve_uploader():
    def collect_files(local_dir):
        """
        Discover HDF5 files for EVE embodiment.
        Call set_directory() first to select the directory.
        """
        hdf5_files = [
            file
            for file in local_dir.iterdir()
            if file.suffix == ".hdf5" and file.is_file()
        ]

        return hdf5_files

    uploader = Uploader(embodiment="eve", datatype=".hdf5", collect_files=collect_files)

    return uploader


def main():
    uploader = eve_uploader()
    asyncio.run(uploader.run())


if __name__ == "__main__":
    main()
