# YOLOv5-GradCAM 

### SKKU S-HERO Capstone Project with PCB Defects detection <br> using YOLOv5 & Grad-CAM (Pytorch)
YOLOv5 Source: https://github.com/ultralytics/yolov5 <br>
Grad-CAM Source: https://github.com/jacobgil/pytorch-grad-cam

# Usage
## 1. Object Detection using YOLOv5
- Check 'yolov5' folder, read usage first
- Structure your files as yolov5's input
- Run train.py for training
- and detect.py to save detection results (if you're not interesting at interpreting results, you can skip from here.)

## 2. Interpret classification results using grad-CAM
### 2-1. Train YOLOv5_classifier
- We're interpreting just classification results, not including detection
- Run train.py for training yolo_classifier: submodel of yolov5 with detecting architecture removed 
- Make sure you use the same structure as yolov5.
- Classification results will be SAME between original yolo and yolo_classifier

### 2-2. Run cam.py
- In 'pytorch-grad-cam' folder
- Modify 'model' to the same model as you used before
- Modify 'ckpt' to your own trained weight
- run script(check pytorch-grad-cam usage)

![A2](https://user-images.githubusercontent.com/75653891/139850763-2acd3026-c134-4232-9d33-ec8dcf417463.jpg)
![KakaoTalk_20211106_204417278](https://user-images.githubusercontent.com/75653891/141776226-da8d69af-ba56-4ee2-9660-97bbc59b2b10.jpg)
![A2_cam](https://user-images.githubusercontent.com/75653891/139850783-f602ac92-06a0-4515-9509-5219091252cf.jpg)

![B2](https://user-images.githubusercontent.com/75653891/139851374-150398c0-d5df-459f-b031-367c1da912a4.jpg)
![KakaoTalk_20211106_204417278_03](https://user-images.githubusercontent.com/75653891/141776325-c2c4efe2-5c3f-4c42-a186-55ad264576b1.jpg)
![B2_cam](https://user-images.githubusercontent.com/75653891/139851381-e9d04366-28d0-4c04-b2dc-986965aec6e0.jpg)

