import cv2
import numpy as np
from PIL import Image


def invert_image_colors(img):
    # 反转颜色
    img = img.point(lambda x: 255 - x)
    return img


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def process_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黄色和白色的颜色范围
    yellow_lower = np.array([14, 65, 120], dtype=np.uint8)
    yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([255, 55, 255], dtype=np.uint8)

    # 创建黄色和白色的遮罩
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # 合并遮罩
    mask = cv2.bitwise_or(yellow_mask, white_mask)

    # 使用形态学操作去除小的噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (0, imshape[0]/2), (imshape[1] / 1.5, imshape[0] / 2),
                         (imshape[1]/1.1, imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(mask, vertices)
    result = np.zeros_like(image)

    # 将遮罩应用于结果图像，将车道部分设置为白色
    result[masked_edges > 0] = [255, 255, 255]

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return result


image = cv2.imread("D:/fourth/dataset2/5.jpg")
img = Image.open("D:/fourth/dataset2/6.png")
img = invert_image_colors(img)
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
output = process_image(image)
result = cv2.bitwise_and(img, output)
cv2.imwrite('D:/fourth/part2/5.jpg', result)
