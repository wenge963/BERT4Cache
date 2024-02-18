docker run -it --gpus all --cpus="40" --rm -v $(pwd):/data -v $(pwd)/ckpt_path:/path --workdir /data tensorflow/tensorflow:2.6.0-gpu /bin/bash /data/train_ml_ckpt-1m.sh
