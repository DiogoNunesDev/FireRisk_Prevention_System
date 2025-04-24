import torch.multiprocessing as mp
from ultralytics import YOLO

if __name__ == "__main__":
  mp.set_start_method('spawn', force=True) 
  model = YOLO("./yolo11m-seg.pt")  
  results = model.train(data="./data.yaml", task='segment', epochs=2000, imgsz=512, device="cuda", batch=1)
