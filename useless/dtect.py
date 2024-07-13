import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import multiprocessing

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Define a function for parallel detection
def detect_frames(chunk, opt):
    source, weights, imgsz = chunk
    device = select_device(opt.device)
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    model = TracedModel(model, device, imgsz)
    cudnn.benchmark = True
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    results = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    results.append((path, xyxy, label, colors[int(cls)]))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)
    
    # Divide video source into chunks
    # Assuming the source is a video file
    cap = cv2.VideoCapture(opt.source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_chunks = multiprocessing.cpu_count()
    chunk_size = total_frames // num_chunks
    chunks = [(opt.source, opt.weights, opt.img_size)] * num_chunks
    
    # Create a pool of processes
    pool = multiprocessing.Pool()
    
    # Map the detection function to each chunk
    results = pool.starmap(detect_frames, [(chunk, opt) for chunk in chunks])
    
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
    
    # Aggregate and save the combined detection results
    # Assuming saving results to a text file
    save_dir = Path('runs/detect')
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'detection_results.txt', 'w') as f:
        for result_list in results:
            for path, xyxy, label, color in result_list:
                f.write(f'{path},{",".join(map(str, xyxy))},{label}\n')
    
    print('Detection done and results saved.')
