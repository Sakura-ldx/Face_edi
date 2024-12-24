import asyncio
import os
import sys
sys.path.append(".")
sys.path.append("..")
import torch
import numpy as np
# import dlib
import cv2
from matplotlib import pyplot as plt
from configs import paths_config
import facer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_aligner = facer.face_aligner('farl/ibug300w/448', device=device)  # optional: "farl/wflw/448", "farl/aflw19/448"


def get_landmark(filepath):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    # image: 1 x 3 x h x w
    image = facer.hwc2bchw(facer.read_hwc(filepath)).to(device=device)

    with torch.inference_mode():
        faces = face_detector(image)
        if not faces:
            return None
        faces = face_aligner(image, faces)

    return faces['alignment']
    # print(faces.keys())
    # img = cv2.imread(filepath)[..., ::-1]
    # vis_img = img.copy()
    # for pts in faces['alignment']:
    #     vis_img = facer.draw_landmarks(vis_img, None, pts.cpu().numpy())
    # cv2.imwrite(os.path.basename(filepath), vis_img)


def modify_smile(ld: torch.Tensor, modify_lambda: float):
    ld[60, 1] = ld[60, 1] + 5 * modify_lambda
    ld[64, 1] = ld[64, 1] + 5 * modify_lambda

    ld[48, 1] = ld[60, 1] + 10 * modify_lambda
    if ld[49, 0] == ld[60, 0]:
        ld[48, 0] = ld[60, 0]
    elif ld[60, 1] == ld[49, 1]:
        ld[48, 0] = ld[60, 0] - 10 * modify_lambda
    else:
        ld[48, 0] = ld[60, 0] - int(10 * modify_lambda / (ld[60, 1] - ld[49, 1]) * (ld[49, 0] - ld[60, 0]))

    ld[54, 1] = ld[64, 1] + 10 * modify_lambda
    if ld[64, 0] == ld[53, 0]:
        ld[54, 0] = ld[53, 0]
    elif ld[64, 1] == ld[53, 1]:
        ld[54, 0] = ld[64, 0] + 10 * modify_lambda
    else:
        ld[54, 0] = ld[64, 0] + int(10 * modify_lambda / (ld[64, 1] - ld[53, 1]) * (ld[64, 0] - ld[53, 0]))

    return ld


def modify_mouth(ld: torch.Tensor, modify_lambda: float):
    # 1. 以48，54直线为基准线，计算对称点与基准线交点
    # 2. 得到交点求平均值得到中心点
    # 3. 49-54，55-60做放射线，找到与原两点连线平行线与放射线交点
    def get_center(points: torch.Tensor):
        return torch.mean(points, dim=0)

    out_center = get_center(ld[48:60, 1])

    ld[60, 1] = ld[60, 1] + 5 * modify_lambda
    ld[64, 1] = ld[64, 1] + 5 * modify_lambda

    ld[48, 1] = ld[60, 1] + 10 * modify_lambda
    if ld[49, 0] == ld[60, 0]:
        ld[48, 0] = ld[60, 0]
    elif ld[60, 1] == ld[49, 1]:
        ld[48, 0] = ld[60, 0] - 10 * modify_lambda
    else:
        ld[48, 0] = ld[60, 0] - int(10 * modify_lambda / (ld[60, 1] - ld[49, 1]) * (ld[49, 0] - ld[60, 0]))

    ld[54, 1] = ld[64, 1] + 10 * modify_lambda
    if ld[64, 0] == ld[53, 0]:
        ld[54, 0] = ld[53, 0]
    elif ld[64, 1] == ld[53, 1]:
        ld[54, 0] = ld[64, 0] + 10 * modify_lambda
    else:
        ld[54, 0] = ld[64, 0] + int(10 * modify_lambda / (ld[64, 1] - ld[53, 1]) * (ld[64, 0] - ld[53, 0]))

    return ld


def draw_landmarks(filepath, modify=None, modify_lambda=1):
    if modify is not None:
        save_dir = os.path.join("../edit_data", f"kp_{modify}_", str(modify_lambda))
        save_ld_dir = os.path.join("../edit_data", f"kp_{modify}_pt_", str(modify_lambda))
    else:
        save_dir = os.path.join("/home/liudongxv/workplace/GANInverter-dev/edit_data", "kp_transform")
        save_ld_dir = os.path.join("/home/liudongxv/workplace/GANInverter-dev/edit_data", "kp_transform_pt")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_ld_dir, exist_ok=True)

    modify_method = None
    if modify == "smile":
        modify_method = modify_smile
    elif modify == "mouth":
        modify_method = modify_mouth

    error = []
    for filename in os.listdir(filepath):
        print(filename)
        read_path = os.path.join(filepath, filename)
        ld = None
        if os.path.splitext(filename)[1] == ".pt":
            ld = torch.load(read_path)
        # else:
        #     ld = get_landmark(read_path)
        #     if ld is None:
        #         error.append(read_path)
        #         continue
        #     ld = ld[0].clone()

        save_path = os.path.join(save_dir, filename)
        # image_origin = cv2.imread(read_path)
        image = np.zeros((1024, 1024), dtype=np.uint8)

        # print(ld.shape)
        if ld is not None:
            if modify_method is not None:
                ld = modify_method(ld, modify_lambda)
            save_ld_path = os.path.join(save_ld_dir, os.path.splitext(filename)[0] + ".pt")
            save_image_path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".jpg")
            torch.save(ld, save_ld_path)

            for x, y in ld:
                cv2.circle(image, (int(x), int(y)), 10, (255, 255, 255), -1)
        cv2.imwrite(save_image_path, image)
    with open(os.path.join(save_dir, "error.txt"), mode='w') as f:
        f.write('\n'.join(error))


if __name__ == '__main__':
    draw_landmarks("/home/liudongxv/workplace/GANInverter-dev/edit_data/kp_pt", modify=None, modify_lambda=2)
    # get_landmark("/nas/Database/Public/CelebA-HQ/total/150992.jpg")


