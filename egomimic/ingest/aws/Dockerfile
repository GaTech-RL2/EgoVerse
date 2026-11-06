# The ingest image: the conversion job (AWS Batch, GPU) and the box's loop run from it.
#   docker build -f egomimic/ingest/aws/Dockerfile --build-arg REVISION=$(git rev-parse HEAD) \
#       --secret id=mano,src=external_ckpts/mano.tar -t egoverse-ingest .
# The MANO model files are license-gated: they come in as a build secret (a tar of
# MANO_LEFT.pkl / MANO_RIGHT.pkl) and stay in the private ECR image, never in git.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git xz-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Static ffmpeg 7 with libsvtav1 (the converter's AV1 encoder).
RUN curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    | tar -xJ -C /opt && ln -s /opt/ffmpeg-*-static/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONPATH=/app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --python 3.11
RUN uv pip install --python /opt/venv/bin/python \
        "mano @ git+https://github.com/otaheri/MANO@5869ab059c1bf31cc724f57eaf93e041135e8960"
COPY egomimic egomimic
ARG REVISION=unknown
RUN echo "$REVISION" > egomimic/ingest/REVISION
RUN --mount=type=secret,id=mano mkdir -p external_ckpts/mano \
    && tar -xf /run/secrets/mano -C external_ckpts/mano
