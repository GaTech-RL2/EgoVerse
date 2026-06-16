import asyncio

from abstract_upload import Uploader


def aria_uploader():
    def collect_files(local_dir):
        """
        Discover VRS files with their corresponding JSON companion files.
        Only processes files that have both .vrs and .vrs.json files present.
        """
        file_paths = []

        vrs_files = [
            file
            for file in local_dir.iterdir()
            if file.suffix == ".vrs" and file.is_file()
        ]

        for vrs_file in vrs_files:
            json_file = vrs_file.with_suffix(f"{vrs_file.suffix}.json")
            if json_file.exists() and json_file.is_file():
                file_paths.append((vrs_file, json_file))

        return file_paths

    uploader = Uploader(
        embodiment="aria",  # Embodiment name
        datatype=".vrs",  # Main data file extension
        collect_files=collect_files,
    )

    return uploader


def main():
    uploader = aria_uploader()
    asyncio.run(uploader.run())


if __name__ == "__main__":
    main()
