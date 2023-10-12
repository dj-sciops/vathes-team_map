# ---------- Download raw ephys/video files ------------
# requires `djsciops` package for fast s3 download (or you can use boto3)
# pip install git+https://github.com/dj-sciops/djsciops-python.git
import os
import datajoint as dj
import boto3
import pathlib
import djsciops.axon as dj_axon


s3_session, s3_bucket = None, None
REMOTE_DIR = r'map_raw_data/behavior_videos'
LOCAL_DIR = r'E:/map/test_data_full'


def _get_s3_session():
    global s3_session, s3_bucket
    if s3_session is None:
        s3_session = boto3.session.Session(
            aws_access_key_id=dj.config['stores']['map_sharing']['access_key'],
            aws_secret_access_key=dj.config['stores']['map_sharing']['secret_key'])
        s3_session.s3 = s3_session.resource("s3")
        s3_bucket = dj.config['stores']['map_sharing']['bucket']
    return s3_session, s3_bucket


SUBFOLDERS = (
    "NewSorting",
    "CompressedVideos/Body",
    "CompressedVideos/Bottom",
    "CompressedVideos/Face",
    "CompressedVideos/other",
    "LightSheet",
    "NewlyCompressed/Body",
    "NewlyCompressed/Bottom",
    "NewlyCompressed/Face",
    "OldSorting"
)


def main():
    _s3_session, _s3_bucket = _get_s3_session()
    for subfolder in SUBFOLDERS:
        print(f"Prepare for download: {subfolder}")
        src = f"{REMOTE_DIR}/{subfolder}"
        dst = pathlib.Path(LOCAL_DIR) / subfolder
        dj_axon.download_files(
            session=_s3_session,
            s3_bucket=_s3_bucket,
            source=f"{src}/",
            destination=f'{dst}{os.sep}',
        )


if __name__ == "__main__":
    main()
