# Face-Mask-Detector
AI model detecting whether people are wearing face mask or not

## Environment
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M57_J3TyLfo_h2Yr915LqS6d_VG6loSU#scrollTo=Gmnfz9jxS8cX)

## Dataset
<img width="500" src="[https://user-images.githubusercontent.com/63842546/213862572-89924584-77c7-448d-b8f8-c8525c66980f.JPG](https://user-images.githubusercontent.com/63842546/214035622-55f90868-8e6d-4875-92ab-10162f8a168a.png)"/>
[Kaggle Dataest](https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset?select=train.csv
)

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
