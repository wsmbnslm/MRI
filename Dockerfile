ARG BASE_CONTAINER

FROM $BASE_CONTAINER as base_stage

RUN mamba install --yes \
    --channel conda-forge \
    'black' \
    'ipympl' \
    'isort' \
    'jupytext' \
	'jupyter-ai' \
    'jupyter-archive' \
    'llama.cpp' \
    'nibabel' \
    'nodejs' \
    'pymupdf' \
    'rclone' \
    'typst' \
    && \
    mamba clean --all -f -y && \
    fix-permissions $CONDA_DIR

RUN pip install --no-cache-dir \
    'jupyterlab-code-formatter' \
    'jupyter-archive' \
    'mrzerocore' \
    'pypulseq' \
    'torch'
    
ENV CLAUDE_CONFIG_DIR=/home/${NB_USER}/.claude

RUN npm install -g \
    --allow-scripts=@agentclientprotocol/claude-agent-acp,@anthropic-ai/claude-code \
    @agentclientprotocol/claude-agent-acp \
    @anthropic-ai/claude-code \
    opencode-ai
