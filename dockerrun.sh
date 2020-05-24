sudo docker run -it --runtime=nvidia --privileged --shm-size=30g -v "$PWD":"$PWD" \
-w "$PWD" \
dgl \
bash

