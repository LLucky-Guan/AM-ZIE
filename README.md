# AM-ZIE  

---

## Introduction  
**A Zero-Reference Illumination Enhancement Model for Complex Underground Mine Environments**

---
## Environment
### Requirements
The requirements can be installed with:
```
torch                          2.0.0+cu118
torchsummary                   1.5.1
torchvision                    0.15.1+cu118
Pillow                         9.4.0
opencv-python                  4.11.0.86
dominate                       2.9.1
einops                         0.8.1
matplotlib                     3.7.1
tqdm                           4.67.1
numpy                          1.24.2
natsort                        8.4.0
lpips                          0.1.4
scikit-image                   0.21.0
```

## Datasets  

| Dataset | Description | Download Link |
|----------|--------------|---------------|
| **LOL** | Low-light image enhancement dataset | [AISTUDIO LOL Dataset](https://aistudio.baidu.com/datasetdetail/119573/1) |
| **SICE** | Multi-exposure image dataset | [AISTUDIO SICE Dataset](https://aistudio.baidu.com/datasetdetail/122364) |
| **DSOD** | drilling site object detection in underground coal mines | [ScienceDB DSOD Dataset](https://doi.org/10.57760/sciencedb.j00001.01020) |

> Please place the datasets in the `./datasets/` directory following the structure described in the code or documentation.

---

##  Model Training  

To train AM-ZIE on your dataset, run the following command:
```
python lowlight_train.py
```
You can modify training parameters such as learning rate, epochs, and batch size in the configuration section of the script.
##  Model Testing

To test the model on the validation or test set, use:
```
python lowlight_test.py
```
Results (enhanced images) will be saved automatically in the ./results/ directory.

