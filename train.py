if __name__ == '__main__':
 from ultralytics import YOLOv10
 model = YOLOv10.from_pretrained('jameslahm/yolov10m')
 model.train(data='coco88.yaml', epochs=500, batch=26, imgsz=640)


# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10m')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

