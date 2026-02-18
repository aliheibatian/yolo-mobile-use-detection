from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
torch.cuda.empty_cache()

def train_model():
    model = YOLO('yolo11m.pt')
    model.train(data='config.yaml', epochs=100,imgsz=640,batch=4,workers=4, device=0)

if __name__ == '__main__':
    freeze_support()
    train_model()