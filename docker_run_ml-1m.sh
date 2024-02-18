docker run -it --privileged --gpus all --cpus="40" --name="run_ml_1m_$(date +%s)" -v $(pwd):/data -v $(pwd)/ckpt_path:/tmp --workdir /data tensorflow/tensorflow:2.6.0-gpu /bin/bash /data/run_ml-1m.sh
