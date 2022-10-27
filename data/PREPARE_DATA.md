# Prepare Data

Download datasets as you need, and organize them as following:
 ```
code_root/
└── data/
    └── coco/
        ├── train2014/
        ├── val2014/
        ├── test2015/
        ├── train2017/
        ├── val2017/
        ├── test2017/
        ├── annotations/
        ├── aokvqa/
            ├── commonsense
                └── expansions
        ├── okvqa/
            ├── commonsense
                └── expansions
        ├── vqa/
        └── vgbua_res101_precomputed/
            ├── trainval2014_resnet101_faster_rcnn_genome
            └── test2015_resnet101_faster_rcnn_genome
        └── sbert/
            ├── aokvqa
            └── okvqa 
        
 ```

## Data

### VQA & RefCOCO+

#### Common
* Download and unzip COCO 2014 images & annotations from [here](http://cocodataset.org/#download).
* For A-OKVQA, we use COCO 2017 images, which you can also download at the above link.

#### VQA
* Download and unzip annotations from [here](https://visualqa.org/download.html) (including "VQA Annotations" and "VQA Input Questions"), 
place all these files directly under ```./data/coco/vqa```.
* Download and unzip following precomputed boxes & features into ```./data/coco/vgbua_res101_precomputed```.
    * train2014 + val2014: [GoogleDrive](https://drive.google.com/file/d/1KyLyqTqBsMX7QtLTma0xFrmhAzdQDUed/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1Udtoi2TC-nAimZf-vLC9PQ)
    * test2015: [GoogleDrive](https://drive.google.com/file/d/10nM3kRz2c827aqwVvLnv430YYFp0po6O/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1wd3rWfPWLBhGkEc10N9e1Q)

* Download answer vocabulary from [GoogleDrive](https://drive.google.com/file/d/1CPnYcOgIOP5CZkp_KChuCg54_Ljr6-fp/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1IvPsH-mmqHi2glgznaBuYw), place it under the folder ```./data/coco/vqa/```.
    
#### OK-VQA
* Download the training and testing files from [here](https://okvqa.allenai.org/download.html) and save them in the `./data/coco/okvqa` folder

#### A-OKVQA 
* Download the training and testing files from [here](https://allenai.org/project/a-okvqa/home) and save them in the `./data/coco/aokvqa` folder 