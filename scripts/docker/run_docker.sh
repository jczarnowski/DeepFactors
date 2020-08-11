xhost +local:
docker run -it \
    --rm \
    --runtime=nvidia \
    -v /home/$(id --user --name):/home/$(id --user --name) \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
    deepfactors
xhost -local: