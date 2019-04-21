detect:
	python .\yolo_video.py --model_path model_data/trained_weights_final.h5 --anchors_path model_data/tiny_yolo_anchors.txt --image --classes_path model_data/vehicle_classes.txt

train:
	python train.py 