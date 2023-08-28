import os
import pathlib
import datajoint as dj

from pipeline import experiment
from pipeline.experiment import get_wr_sessdatetime
from pipeline.export.nwb import export_recording

#dj.conn().set_query_cache('s0')

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

# ---------- Download raw ephys files ------------
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


# ---------- Download raw video files ------------
# from mounted Google Drive


VIDEO_LOCAL_DIR = pathlib.Path(dj.config['custom']['tracking_data_paths'][0])
VIDEO_REMOTE_DIR = pathlib.Path(r'G:/.shortcut-targets-by-id/1fNJy5IkEqUZJUaB0KHs1cLiuTguvxgFr/Compressed MAP Videos')


def download_raw_video(session_key):
    import shutil

    wr, _ = get_wr_sessdatetime(session_key)
    sess_date, sess_time = (experiment.Session & session_key).fetch1(
        'session_date', 'session_time')
    sess_date_str = sess_date.strftime('%m%d%y')

    srcs = ([d for d in VIDEO_REMOTE_DIR.glob(f'*/{wr}/{wr}_{sess_date_str}')
             if d.is_dir()]
            + [d for d in VIDEO_REMOTE_DIR.glob(f'*/{wr}_{sess_date_str}')
               if d.is_dir()])
    dsts = [VIDEO_LOCAL_DIR / d.relative_to(VIDEO_REMOTE_DIR) for d in srcs]

    for src, dst in zip(srcs, dsts):
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
