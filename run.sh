export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d '=' -f2)
docker run \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -v $HOME/.anthropic:/home/computeruse/.anthropic \
    -v ./computer_use_demo:/home/computeruse/computer_use_demo \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
