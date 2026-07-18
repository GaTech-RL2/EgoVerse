import argparse
import asyncio

from abstract_upload import Uploader

# Robot configurations the aria hardware can record. This is the metadata
# `embodiment` (→ SQL embodiment column), NOT the capture device ("aria").
ARIA_EMBODIMENTS = ("human_bimanual", "human_left_arm", "human_right_arm")


def aria_uploader(embodiment="human_bimanual"):
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
        embodiment=embodiment,  # robot config recorded → metadata.embodiment
        datatype=".vrs",  # Main data file extension
        collect_files=collect_files,
        device="aria",  # raw capture source → s3://rldb/raw_v2/aria/
    )

    return uploader


def main():
    parser = argparse.ArgumentParser(
        description="Upload raw aria (.vrs) recordings to s3://rldb/raw_v2/aria/."
    )
    parser.add_argument(
        "--embodiment",
        default="human_bimanual",
        choices=ARIA_EMBODIMENTS,
        help="Robot embodiment recorded by the aria hardware; written to "
        "metadata.embodiment (the raw S3 path stays raw_v2/aria/ regardless). "
        "Default: human_bimanual.",
    )
    args = parser.parse_args()
    uploader = aria_uploader(embodiment=args.embodiment)
    asyncio.run(uploader.run())


if __name__ == "__main__":
    main()
