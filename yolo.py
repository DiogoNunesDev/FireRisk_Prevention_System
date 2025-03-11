import torch.multiprocessing as mp
from ultralytics import YOLO

if __name__ == "__main__":
  mp.set_start_method('spawn', force=True)  # <- Add this
  model = YOLO("./yolo11n-seg.pt")  
  results = model.train(data="./data.yaml", epochs=2000, imgsz=512, device="cuda")
