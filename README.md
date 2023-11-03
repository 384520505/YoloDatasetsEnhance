# YoloDatasetsEnhance
### yolo数据增强、yolo已标注数据集增强、.txt格式数据集增强；包含旋转、平移、翻转、裁剪、调整亮度和增加噪声6中增强方式随机组合，每张图片扩增到5张。

### 运行环境
- 系统：MAC
- Conda虚拟环境软件包配置：
- Package             Version
- ------------------- ---------
- imgaug              0.4.0
- numpy               1.24.3
- opencv-python       4.1.2.30

* 注意：opencv-python版本不能太高，否则报错 *

  ### 使用教程
  - 如果已标注的数据集格式为txt，将其转为xml格式，运行TxtTransfromXml.py文件
```python
# 修改原图片和txt文件路径
if __name__ == '__main__':
    # Volumes/ACASIS_Media/DeepLearning/ultralytics/datasets/preson
    root_path = r'datasets/tea/images/val'  # 图片路径
    txt_root = r'datasets/tea/labels/val'   # txt路径

    gt_labels = ['tea']

    yolov5txt2xml(root_path, txt_root, gt_labels=gt_labels)

#修改xml文件存放的路径
out_root = r'datasets/tea/labels' # xml文件存放路径
```
 - 获取到xml格式的数据集后，对其进行增强，运行datasets_engine.py文件，进行增强
 - 将增强后的xml格式的文件转为txt格式，运行XmlTransfromTxt.py文件
```python
# 修改路径
image_set = 'datasets/tea/labels/val_xml_enhance_2'  # 需要转换的文件夹名称（文件夹内放xml标签文件）
imageset2 = 'datasets/tea/labels/val_txt_enhance_2'  # 保存txt的文件夹，需要先创建val_txt_enhance_2文件夹，否则运行会报错
```
