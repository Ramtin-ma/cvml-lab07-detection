# assets/

Download pretrained detector here as `yolov5s.onnx` (input 1x3x640x640, output 1x25200x85).
https://my.sharif.edu/s/kzNyRYGxKFkrmWa

How to produce it (instructor, once):

```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
python export.py --weights yolov5s.pt --include onnx --imgsz 640 --opset 12
```