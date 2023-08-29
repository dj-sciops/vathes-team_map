import os
import pathlib
import datajoint as dj

from pipeline.export.nwb import export_recording, DEV_POS_FOLDER_MAPPING
from pipeline.export.nwb import experiment, tracking, tracking_ingest
from pipeline.experiment import get_wr_sessdatetime

output_dir = pathlib.Path(r'E:/map/NWB_EXPORT/delay_response')

project_name = 'Brain-wide neural activity underlying memory-guided movement'


def export_to_nwb(limit=None):
    session_keys = (experiment.Session & (experiment.ProjectSession
                                          & {'project_name': project_name})).fetch(
        'KEY', limit=limit)

    for session_key in session_keys:
        # session_key = {"subject_id": 440959, 'session': 5}
        download_raw_ephys(session_key)
        download_raw_video(session_key)
        export_recording(session_key, output_dir=output_dir,
                         overwrite=False, validate=False,
                         raw_ephys=True, raw_video=True)


dandiset_id = os.getenv('DANDISET_ID', dj.config['custom'].get('DANDISET_ID'))
dandi_api_key = os.getenv('DANDI_API_KEY', dj.config['custom'].get('DANDI_API_KEY'))


def publish_to_dandi(dandiset_id, dandi_api_key):
    from element_interface.dandi import upload_to_dandi

    dandiset_dir = output_dir.parent / f"{output_dir.name}_DANDI"
    dandiset_dir.mkdir(parents=True, exist_ok=True)

    upload_to_dandi(
        data_directory=output_dir,
        dandiset_id=dandiset_id,
        staging=False,
        working_directory=dandiset_dir,
        api_key=dandi_api_key,
        sync=False,
        existing='overwrite')

# ---------- Download raw ephys/video files ------------
# requires `djsciops` package for fast s3 download (or you can use boto3)
# pip install git+https://github.com/dj-sciops/djsciops-python.git

import datajoint as dj
import boto3
import djsciops.axon as dj_axon

s3_session = boto3.session.Session(
    aws_access_key_id=dj.config['stores']['map_sharing']['access_key'],
    aws_secret_access_key=dj.config['stores']['map_sharing']['secret_key'])
s3_session.s3 = s3_session.resource("s3")
s3_bucket = dj.config['stores']['map_sharing']['bucket']

EPHYS_LOCAL_DIR = pathlib.Path(dj.config['custom']['ephys_data_paths'][0])
EPHYS_REMOTE_DIR = r'map_raw_data/behavior_videos/NewSorting'


def download_raw_ephys(session_key):
    wr, _ = get_wr_sessdatetime(session_key)
    sess_date, sess_time = (experiment.Session & session_key).fetch1(
        'session_date', 'session_time')
    sess_date_str = sess_date.strftime('%m%d%y')

    sess_relpath = f"{wr}_out/results/catgt_{wr}_{sess_date_str}_g0"

    sess_remote_path = f"{EPHYS_REMOTE_DIR}/{sess_relpath}/"
    sess_local_path = EPHYS_LOCAL_DIR / sess_relpath

    file_list = dj_axon.list_files(
        session=s3_session,
        s3_bucket=s3_bucket,
        s3_prefix=sess_remote_path,
        permit_regex=r".*\.ap\.(meta|bin)",
        include_contents_hash=False,
        as_tree=False,
    )
    total_gb = sum(f['_size'] for f in file_list) * 1e-9
    print(f"Total GB to download: {total_gb}")
    dj_axon.download_files(
        session=s3_session,
        s3_bucket=s3_bucket,
        source=sess_remote_path,
        destination=f'{sess_local_path}{os.sep}',
        permit_regex=r".*\.ap\.(meta|bin)",
    )

    return [sess_local_path]


VIDEO_LOCAL_DIR = pathlib.Path(dj.config['custom']['tracking_data_paths'][0])
VIDEO_REMOTE_DIR = r'map_raw_data/behavior_videos/CompressedVideos'


def download_raw_video(session_key):
    wr, _ = get_wr_sessdatetime(session_key)
    tracking_positions = (tracking.TrackingDevice
                          & (tracking.Tracking & session_key)).fetch('tracking_position')
    _one_file = (tracking_ingest.TrackingIngest.TrackingFile
                 & session_key).fetch('tracking_file', limit=1)[0]
    sess_dir_name = pathlib.Path(_one_file).parts[1]

    sess_local_paths = []
    for trk_pos in tracking_positions:
        camera_str = DEV_POS_FOLDER_MAPPING[trk_pos]
        sess_remote_path = f"{VIDEO_REMOTE_DIR}/{camera_str}/{sess_dir_name}/"
        sess_local_path = VIDEO_LOCAL_DIR / camera_str / wr / f"{sess_dir_name}"

        file_list = dj_axon.list_files(
            session=s3_session,
            s3_bucket=s3_bucket,
            s3_prefix=sess_remote_path,
            permit_regex=r".*\.mp4",
            include_contents_hash=False,
            as_tree=False,
        )
        total_gb = sum(f['_size'] for f in file_list) * 1e-9
        print(f"Total GB to download: {total_gb}")
        dj_axon.download_files(
            session=s3_session,
            s3_bucket=s3_bucket,
            source=sess_remote_path,
            destination=f'{sess_local_path}{os.sep}',
            permit_regex=r".*\.mp4",
        )

        sess_local_paths.append(sess_local_path)

    return sess_local_paths
