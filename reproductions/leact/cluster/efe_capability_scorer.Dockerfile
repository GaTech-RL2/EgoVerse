FROM docker.io/library/python@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199
RUN python -m pip install --no-cache-dir numpy==1.26.4 pyarrow==21.0.0 pandas==2.2.3 huggingface-hub==0.28.1 datasets==3.3.2 wget==3.2 tempdir==0.7.1 multipledispatch==1.0.0 appdirs==1.4.4 tqdm==4.67.1 termcolor==3.1.0 fire==0.7.0 rich==14.0.0 tree-sitter==0.23.2 tree-sitter-python==0.23.6 psutil==7.0.0 absl-py==2.3.1 langdetect==1.0.9 nltk==3.9.2 immutabledict==4.2.2
RUN python -m nltk.downloader -d /opt/nltk punkt punkt_tab
COPY vendor /opt/vendor
COPY score.py /opt/score.py
ENV PYTHONPATH=/opt/vendor:/opt/vendor/evalplus NLTK_DATA=/opt/nltk XDG_CACHE_HOME=/tmp/cache HF_HOME=/tmp/hf PYTHONDONTWRITEBYTECODE=1 HUMANEVAL_OVERRIDE_PATH=/inputs/humaneval.jsonl
RUN python -m pip freeze > /opt/packages.txt
RUN python -c "from evalplus.sanitize import sanitize; from evalplus.evaluate import evaluate; from instruction_following_eval import evaluation_lib"
