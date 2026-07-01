ARG BASE_CONTAINER

FROM $BASE_CONTAINER as base_stage

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
