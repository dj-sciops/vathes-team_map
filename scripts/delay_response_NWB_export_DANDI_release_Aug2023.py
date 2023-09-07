"""
Additional packages required

pip install git+https://github.com/dj-sciops/djsciops-python.git
pip install --upgrade pyopenssl
pip install opencv-python
pip install git+https://github.com/ttngu207/element-interface.git

apt-get install ffmpeg libsm6 libxext6 g++ -y
"""

from pipeline import publication


populate_settings = dict(reserve_jobs=True, suppress_errors=True)


def main():
    for _ in range(6):
        _clean_up()
        publication.DANDIupload.populate(max_calls=1, **populate_settings)
        publication.NWBFileExport.populate(max_calls=1, **populate_settings)
        publication.DANDIupload.populate(max_calls=1, **populate_settings)


def _clean_up():
    _generic_errors = [
        "%Deadlock%",
        "%Lock wait timeout%",
        "%MaxRetryError%",
        "%KeyboardInterrupt%",
        "InternalError: (1205%",
        "%SIGTERM%",
        "%LostConnectionError%",
    ]
    (publication.schema.jobs
     & 'status = "error"'
     & [f'error_message LIKE "{e}"' for e in _generic_errors]
     ).delete()


if __name__ == "__main__":
    main()
