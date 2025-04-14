# Computer use setup

- Setup logging for computer use
- All logs (with screenshots) saved to `./conversation_logs`

## Setup
- Docker should be setup and running.
- `sh run.sh` should start the Computer Use instance at [localhost:8080](https://localhost:8080).

## Implementation information
- line 5 
`-v ./computer_use_demo:/home/computeruse/computer_use_demo`
mounts the specified folder to the docker instance.
