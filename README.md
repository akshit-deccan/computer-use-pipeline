# Computer use setup

Makes it easy to setup logging and tweak other options when 
using Anthropic Computer Use.

- All files in `computer\_use_demo/`.
- Most work would be in `streamlit.py`.

## Setup
- Docker should be setup and running.
- `sh run.sh` should start the Computer Use instance at [localhost:8080](https://localhost:8080).

## Implementation information
- line 5 
`-v ./computer_use_demo:/home/computeruse/computer_use_demo`
mounts the specified folder to the docker instance, for logging, files would be saved inside the folder itself.
