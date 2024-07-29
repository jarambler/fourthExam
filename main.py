import cv2
import numpy as np
from PIL import Image


def match_images(img2, img1):
    # 检查图像是否成功加载
    if img2 is None or img1 is None:
        raise IOError("Could not read one or both images")

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用FLANN进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 只保留好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 100:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用RANSAC算法估计仿射变换矩阵
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            return None

        # 获取图像的尺寸
        h1, w1 = img1.shape
        h2, w2 = img2.shape

        # 画出匹配区域的矩形框
        pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        dst = cv2.transform(pts, M)
        img1_with_box = cv2.polylines(img1.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # 创建掩码
        mask = np.zeros_like(img1)
        cv2.fillPoly(mask, [np.int32(dst)], 255)
        img1_with_box_black = cv2.bitwise_and(img1_with_box, mask)

        # 进行SIFT特征检测和匹配
        kp1, des1= sift.detectAndCompute(img1_with_box_black, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 绘制匹配结果
        img_matches = cv2.drawMatches(img1_with_box_black, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        image = np.tile(np.array(img1_with_box)[..., np.newaxis], (1, 1, 3))
        image = Image.fromarray(np.uint8(image))
        new_img = Image.new('RGB', (w2 + w1, h1), (0, 0, 0))
        new_img.paste(image, (0, 0, min(new_img.size[0], image.size[0]), min(new_img.size[1], image.size[1])))
        img_array = np.array(new_img)
        img_final = img_matches | img_array

        # 加深匹配线
        for match in matches[:10]:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2) + img1_with_box_black.shape[1], int(y2)
            cv2.line(img_final, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img_final

    else:
        return None


def adjust_contrast_and_brightness(image, contrast=1.0, brightness=0):

    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


# 读取图像并进行匹配
for i in range(1, 46):
    query_img = cv2.imread("D:/fourth/dataset3/archive/{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
    contrast = 3  # 对比度值
    brightness = -20  # 亮度值
    query_img = adjust_contrast_and_brightness(query_img, contrast, brightness)
    templates = [
        cv2.imread("D:/fourth/dataset3/template/template_1.jpg", cv2.IMREAD_GRAYSCALE),
        cv2.imread("D:/fourth/dataset3/template/template_2.jpg", cv2.IMREAD_GRAYSCALE),
        cv2.imread("D:/fourth/dataset3/template/template_3.jpg", cv2.IMREAD_GRAYSCALE)
    ]
    adjusted_templates = []
    for template in templates:
        adjusted_template = adjust_contrast_and_brightness(template, contrast, brightness)
        adjusted_templates.append(adjusted_template)

    for j, template_img in enumerate(adjusted_templates):
        matched_img = match_images(template_img, query_img)
        if matched_img is not None:
            cv2.imwrite(f"D:/fourth/part3/{i}-{j+1}.jpg", matched_img)
        else:
            print("Not enough matches are found")
