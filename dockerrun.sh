sudo docker run -it --runtime=nvidia --privileged --shm-size=30g -v "$PWD":"$PWD" \
-v '/media/ddangelo/External1/geosim':'/geosim' \
-w "$PWD" \
dgl \
bash

