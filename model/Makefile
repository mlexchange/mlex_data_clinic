TAG 			:= latest	
USER 			:= mlexchange
PROJECT			:= unsupervised-classifier

IMG_WEB_SVC    		:= ${USER}/${PROJECT}:${TAG}
IMG_WEB_SVC_JYP    	:= ${USER}/${PROJECT_JYP}:${TAG}
ID_USER			:= ${shell id -u}
ID_GROUP			:= ${shell id -g}

.PHONY:

test:
	echo ${IMG_WEB_SVC}
	echo ${TAG}
	echo ${PROJECT}
	echo ${PROJECT}:${TAG}
	echo ${ID_USER}

build_docker: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile .

run_docker:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v ${PWD}:/app/work/ -v /data/tanchavez/Datasets/scattering:/app/work/data -p 8888:8888 ${IMG_WEB_SVC}

train_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/scattering:/app/work/data ${IMG_WEB_SVC} python3 src/train_model.py data/mixed_small_32x32.npz data/out '{"latent_dim": 16, "shuffle": true, "batch_size": 32, "num_epochs": 1, "base_channel_size": 32, "validation_ok": false}' 

evaluate_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/born:/app/work/data ${IMG_WEB_SVC} python3 src/evaluate_model.py data/test data/logs/model.h5 '{"rotation_angle": 0, "image_flip": "None", "batch_size": 1}'

predict_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/born:/app/work/data ${IMG_WEB_SVC} python3 src/predict_model.py data/test data/logs/model.h5 data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 1}'

transfer_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/born:/app/work/data ${IMG_WEB_SVC} python3 src/transfer_learning.py data/train data/val data/logs/model.h5 data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32, "pooling": "None", "epochs": 10, "init_layer": 100}'


clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache

push_docker:
	docker push ${IMG_WEB_SVC}
