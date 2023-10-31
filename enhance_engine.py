# 数据增强
import os
import cv2
import numpy as np

# 输入文件夹和输出文件夹
src_img_dir = '../datasets/stars/valid/images' # 原始图片目录
src_txt_dir = '../datasets/stars/valid/labels' # 原始标注目录
output_img_dir = '../datasets/stars/validHot/images' # 输出图片目录
output_txt_dir = '../datasets/stars/validHot/labels' # 输出标注目录

# 创建输出文件夹
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_txt_dir, exist_ok=True)

# 随机生成图像增强参数范围
translation_range = np.random.randint(10, 50)  # 随机生成平移范围
rotation_angle_range = np.random.randint(10, 30)  # 随机生成旋转角度范围
crop_range = np.random.randint(10, 30)  # 随机生成裁剪范围
brightness_range = (np.random.uniform(0.5, 1.5), np.random.uniform(1.5, 2.5))  # 随机生成亮度调节范围
noise_stddev = np.random.randint(5, 20)  # 随机生成噪声标准差

# 获取图像文件和标注文件的路径
img_paths = [os.path.join(src_img_dir, f) for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
txt_paths = [os.path.join(src_txt_dir, f) for f in os.listdir(src_txt_dir) if f.endswith('.txt')]

for img_path, txt_path in zip(img_paths, txt_paths):
    # 读取图像
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    # 读取标注数据
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()

    augmentations = np.random.choice(['mirror', 'translate', 'rotate', 'brightness', 'noise', 'crop', 'flip'], size=5, replace=False)

    for i, aug in enumerate(augmentations):
        image_augmented = image.copy()
        lines_augmented = list(lines)

        if aug == 'mirror':
            image_augmented = cv2.flip(image_augmented, 1)
            for j, line in enumerate(lines_augmented):
                values = line.strip().split(' ')
                if len(values) == 5:
                    x, y, width, height = map(float, values[1:5])
                    x = 1 - x
                    lines_augmented[j] = f"{values[0]} {x} {values[2]} {values[3]} {values[4]}\n"

        if aug == 'translate':
            tx = np.random.randint(-translation_range, translation_range)
            ty = np.random.randint(-translation_range, translation_range)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image_augmented = cv2.warpAffine(image_augmented, M, (w, h))

            for j, line in enumerate(lines_augmented):
                values = line.strip().split(' ')
                if len(values) == 5:
                    x, y, width, height = map(float, values[1:5])
                    x += tx / w
                    y += ty / h
                    lines_augmented[j] = f"{values[0]} {x} {y} {values[2]} {values[3]}\n"

        if aug == 'rotate':
            angle = np.random.randint(-rotation_angle_range, rotation_angle_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image_augmented = cv2.warpAffine(image_augmented, M, (w, h))

            for j, line in enumerate(lines_augmented):
                values = line.strip().split(' ')
                if len(values) == 5:
                    x, y, width, height = map(float, values[1:5])
                    x_rot, y_rot = np.dot(M, np.array([x * w, y * h, 1]))
                    x_rot /= w
                    y_rot /= h
                    lines_augmented[j] = f"{values[0]} {x_rot} {y_rot} {values[2]} {values[3]}\n"

        if aug == 'brightness':
            brightness_factor = np.random.uniform(*brightness_range)
            image_augmented = cv2.convertScaleAbs(image_augmented, alpha=brightness_factor, beta=0)

        if aug == 'noise':
            noise = np.random.normal(0, noise_stddev, image_augmented.shape).astype(np.uint8)
            image_augmented = cv2.add(image_augmented, noise)

        if aug == 'crop':
            x1 = np.random.randint(0, crop_range)
            y1 = np.random.randint(0, crop_range)
            x2 = w - np.random.randint(0, crop_range)
            y2 = h - np.random.randint(0, crop_range)
            image_augmented = image_augmented[y1:y2, x1:x2]
            for j, line in enumerate(lines_augmented):
                values = line.strip().split(' ')
                if len(values) == 5:
                    x, y, width, height = map(float, values[1:5])
                    x = (x - x1) / (x2 - x1)
                    y = (y - y1) / (y2 - y1)
                    lines_augmented[j] = f"{values[0]} {x} {y} {values[2]} {values[3]}\n"

        if aug == 'flip':
            image_augmented = cv2.flip(image_augmented, 0)
            for j, line in enumerate(lines_augmented):
                values = line.strip().split(' ')
                if len(values) == 5:
                    y = 1 - float(values[2])
                    lines_augmented[j] = f"{values[0]} {values[1]} {y} {values[2]} {values[3]}\n"

        # 保存增强后的图像
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(output_img_dir, f"{i}_{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(output_img_path, image_augmented)

        # 保存增强后的标注数据
        txt_name = os.path.basename(txt_path)
        output_txt_path = os.path.join(output_txt_dir, f"{i}_{txt_name}")
        with open(output_txt_path, 'w') as output_txt_file:
            output_txt_file.writelines(lines_augmented)

print("数据增强完成")

