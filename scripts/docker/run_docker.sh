# You can change this mount directory to your environment.
DATA_PATH="/home/$(id --user --name)"

xhost +local:
docker run -it \
    --rm \
    --runtime=nvidia \
    -v ${DATA_PATH}:${DATA_PATH} \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
    deepfactors
xhost -local: