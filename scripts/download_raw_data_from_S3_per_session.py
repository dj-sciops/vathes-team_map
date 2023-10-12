# ---------- Download raw ephys/video files ------------
# requires `djsciops` package for fast s3 download (or you can use boto3)
# pip install git+https://github.com/dj-sciops/djsciops-python.git
from pipeline.publication import NWBFileExport, download_raw_ephys, download_raw_video


unfinished_sessions_only = True


def main():
    unfinished_sessions = (NWBFileExport.key_source - NWBFileExport).fetch('KEY')
    finished_sessions = (NWBFileExport.key_source & NWBFileExport).fetch('KEY')

    for key in unfinished_sessions:
        print(f"Downloading raw data for: {key}")
        download_raw_ephys(key)
        download_raw_video(key)

    if unfinished_sessions_only:
        return

    for key in finished_sessions:
        print(f"Downloading raw data for: {key}")
        download_raw_ephys(key)
        download_raw_video(key)


if __name__ == "__main__":
    main()
