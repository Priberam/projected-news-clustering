# DistilReprService

This code uses the [sentence_transformers library](https://github.com/UKPLab/sentence-transformers).


Docker for launching standalone distil server.

Helper script to monitor memory usage:
while sleep 1; do ps -aux | grep -E "[t]ornado"; done


In order to run the docker with CPU, run `sudo docker-compose -f docker_composes/docker_cpu_py36-compose.yml up -d`

To run the docker with the GPU, run `sudo docker-compose -f docker_composes/docker_cuda11-2_py36-compose.yml up -d`

The GPU docker depends on the nvidia/cuda docker; to use it, run the command:
`docker pull nvidia/cuda`
