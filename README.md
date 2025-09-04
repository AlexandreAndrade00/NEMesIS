# Neuroevolution of Efficient Models for Image Segmentation (NEMesIS)

## Installation

Use your favorite python environment tool and install the packages in `requirements.txt`

### pip3 example

```sh
python3 -m venv .venv

source .venv/bin/activate

python3 -m pip install -r requirements.txt
```

## Usage

`-c` `[mandatory]`: sets the path to the config file to be used (in yaml or json format).

`-g` `[mandatory]`: path to the grammar file to be used.

`-r` `[mandatory]`: identifies the run id and seed to be used;

`--gpu-enabled` `[optional]`: if this flag is enabled, the run is performed in a gpu

### Shell
```sh
python3 -m nemesis.main
    -c <config_path>
    -g <grammar_path>
    -r <run>
    --gpu-enabled
```

### VSCode launch
```json
{
    "name": "NEMesIS",
    "type": "debugpy",
    "request": "launch",
    "module": "nemesis.main",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "args": [
        "-c",
        "${cwd}/settings/configs/enc_dec.json",
        "-g",
        "${cwd}/settings/grammars/enc_dec.grammar",
        "--gpu-enabled",
        "-r",
        "${command:pickArgs}"
    ]
}
```

### Docker (Dockerfile for Nvidia, can be easily changed for AMD)

It is necessary to create a volume named `datasets` to save the used datasets, some may require manual download.

```sh
docker build -t nemesis .

docker run -v datasets:/usr/local/app/data --ipc=host --gpus all -it nemesis -c <config_path> -g <grammar_path> -r <run> --gpu-enabled
```

