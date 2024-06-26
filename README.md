# GarbadgeDetection

### Dataset
- UAVVaste drone dataset, source: https://github.com/PUTvision/UAVVaste
- TACO Open trash in wild dataset, source: http://tacodataset.org/
- Cigarette Butts dataset, source: https://www.immersivelimit.com/datasets/cigarette-butts

### Papers
- Xia, Z., Zhou, H., Yu, H. et al. YOLO-MTG: a lightweight YOLO model for multi-target garbage detection. SIViP (2024). https://doi.org/10.1007/s11760-024-03220-2
- Shoufeng Jin, Zixuan Yang, Grzegorz Królczykg, Xinying Liu, Paolo Gardoni, Zhixiong Li,
Garbage detection and classification using a new deep learning-based machine vision system as a tool for sustainable waste recycling,Waste Management,Volume 162,2023,Pages 123-130,ISSN 0956-053X,https://doi.org/10.1016/j.wasman.2023.02.014.

# Requirements
You can install these dependencies using the following pip command in your terminal:

```bash
$ pip install -r requirements.txt # Installs all the required packages
```

# Download Dataset
To download TACO dataset, run download script in your terminal:

```bash
$ python download.py # Installs TACO dataset
```

# Convert Dataset
To convert TACO dataset from COCO format to YOLO format, run JSON2YOLO script in your terminal:
```bash
$ cd JSON2YOLO
$ python coco2yolo.py
```