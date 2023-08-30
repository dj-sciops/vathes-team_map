"""
Additional packages required

pip install git+https://github.com/dj-sciops/djsciops-python.git
pip install --upgrade pyopenssl
pip install opencv-python
pip install git+https://github.com/ttngu207/element-interface.git

apt-get install ffmpeg libsm6 libxext6 g++ -y
"""

from pipeline import publication


populate_settings = dict(max_calls=1, reserve_jobs=True, suppress_errors=True)


def main():
    keys = publication.NWBFileExport.key_source.fetch('KEY')

    key = keys[6]

    publication.NWBFileExport.populate(key, **populate_settings)
    publication.DANDIupload.populate(key, **populate_settings)


if __name__ == "__main__":
    main()
