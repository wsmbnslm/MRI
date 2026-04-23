# Magnetic Resonance Imaging

## Prerequisites

- Podman (https://podman.io/)

### Prerequisites (Windows specific)

- Git for Windows (https://git-scm.com)

## Setup (Windows specific)

### Podman on Windows using WSL2

For Podman on Windows, `Virtual Machine feature` needs to be enabled (using `PowerShell` with admin rights)

```
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

and `Windows Subsystem for Linux 2 (WSL2)` needs to be up-to-dated (again using `PowerShell` with admin rights)

```
wsl --update
```

Afterwards set up Podman. Please note that all *following* commands should be run from a `bash` shell (such as `Git Bash`) not `PowerShell`.

### NVIDIA GPU Support

For NVIDIA GPU support, follow instructions (https://podman-desktop.io/docs/podman/gpu).

First, connect to Podman machine

```
podman machine ssh
```

and then install NVIDIA Container Toolkit onto the Podman Machine

```
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  tee /etc/yum.repos.d/nvidia-container-toolkit.repo && \
  yum install -y nvidia-container-toolkit && \
  nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && \
  nvidia-ctk cdi list
```

## SSH Key

Create ssh-key with

```
ssh-keygen
```

and then register public key in `git.fh-aachen.de`.

Then set git username and e-mail with

```
git config --global user.name "John Doe"
git config --global user.email john.doe@e-mail.com
```

## Build Docker Image Locally

Log into registry

```
podman login registry.git.fh-aachen.de
```

### Build CPU Version

Build locally

```
podman build \
    --tag registry.git.fh-aachen.de/lmp/mri/mri-lab \
    --build-arg BASE_CONTAINER=quay.io/jupyter/datascience-notebook:latest \
    --build-arg BUILD=cpu \
    --file Dockerfile
```

Push to remote

```
podman push registry.git.fh-aachen.de/lmp/mri/mri-lab
```

### Build CUDA Version

Build locally

```
podman build \
    --tag registry.git.fh-aachen.de/lmp/mri/mri-lab-cuda \
    --build-arg BASE_CONTAINER=quay.io/jupyter/datascience-notebook:latest \
    --build-arg BUILD=cuda \
    --build-arg CUDA_VERSION=13.0 \
    --file Dockerfile
```

Push to remote

```
podman push registry.git.fh-aachen.de/lmp/mri/mri-lab-cuda
```

## Setup Using Docker Image (Linux)

Check if container is running

```
podman ps -a
```

If status is not up, run

```
podman start mri
```

If container does not exist, start it with

```
podman run \
  --name mri \
  --detach \
  --userns keep-id:uid=1000,gid=100 \
  --user root \
  --env GRANT_SUDO=yes \
  --volume $HOME/.config/rclone:/home/jovyan/.config/rclone:Z \
  --volume $HOME/.jupyter:/home/jovyan/.jupyter:Z \
  --volume $HOME/.ssh:/home/jovyan/.ssh:Z \
  --volume $HOME/Documents/mri:/home/jovyan/mri:Z \
  --volume $HOME/Documents/mri-data:/home/jovyan/mri-data:Z \
  --publish 8888:8888 \
  registry.git.fh-aachen.de/lmp/mri/mri-lab:latest \
  start-notebook.sh --NotebookApp.token=""
```

and connect to it via `http://localhost:8888`

## Setup Using Docker Image (Windows, WIP)

Check if container is running

```
podman ps -a
```

If status is not up, run

```
podman start mri
```

If container does not exist, start it with

```
podman run \
  --name mri \
  --detach \
  --userns=keep-id:uid=1000,gid=100 \
  --user root \
  --env GRANT_SUDO=yes \
  --volume C:\\Documents\\mri:/home/jovyan/mri:Z \
  --volume C:\\Documents\\mri-data:/home/jovyan/mri-data:Z \
  --publish 8888:8888 \
  --gpus all \
  registry.git.fh-aachen.de/lmp/mri/mri-lab:latest \
  start-notebook.sh --NotebookApp.token=""
```
