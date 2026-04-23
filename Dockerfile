ARG BASE_CONTAINER
ARG BUILD

FROM $BASE_CONTAINER as base_stage

ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}

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
    'mrzerocore'

FROM base_stage AS cpu_stage

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    'torch'

FROM base_stage AS cuda_stage

ARG CUDA_VERSION

RUN CUDA_VERSION_NO_DOTS=$(echo "$CUDA_VERSION" | tr -d '.') && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_NO_DOTS} \
    'torch'

FROM ${BUILD}_stage AS final_stage
