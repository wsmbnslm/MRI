ARG BASE_CONTAINER

FROM $BASE_CONTAINER as base_stage

USER $ROOT

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        lmodern \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

RUN mamba install --yes \
    --channel conda-forge \
    'black' \
    'ipympl' \
    'isort' \
    'nibabel' \
    'rclone' \
    && \
    mamba clean --all -f -y && \
    fix-permissions $CONDA_DIR

RUN pip install --no-cache-dir \
    'jupyterlab-code-formatter' \
    'jupyter-archive' \
    'pypulseq' \
    'pytorch_finufft' \
    'mrzerocore'
