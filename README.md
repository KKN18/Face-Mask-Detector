# Face-Mask-Detector
AI model detecting whether people are wearing face mask or not

## Environment
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M57_J3TyLfo_h2Yr915LqS6d_VG6loSU#scrollTo=Gmnfz9jxS8cX)

## Dataset
<img width="500" src="https://user-images.githubusercontent.com/63842546/214035622-55f90868-8e6d-4875-92ab-10162f8a168a.png"/>
[Kaggle Dataest](https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset?select=train.csv)

## Model
### Predict face mask and Create bounding box

```python
num_classes = 3
model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
```
#### Faster R-CNN
<img width="300" src="https://user-images.githubusercontent.com/63842546/214036050-8b37f3dd-e75f-4d70-bc39-421ccca56dc6.png"/>

## Result
### Input Image #1
<img width="300" src="https://user-images.githubusercontent.com/63842546/214036488-d4e2548d-5d12-441a-b33e-75229fa483c7.png"/>

### Output Image #1
<img width="300" src="https://user-images.githubusercontent.com/63842546/214036772-21550621-3aeb-4588-bb15-a97076c24a19.png"/>

### Input Image #2
<img width="300" src="https://user-images.githubusercontent.com/63842546/214037050-eb3dd636-c6ee-436b-9624-25debf1a3172.png"/>

### Output Image #2
<img width="300" src="https://user-images.githubusercontent.com/63842546/214037093-95aab6b4-9606-42d1-808d-4d65d4dedf1e.png"/>

