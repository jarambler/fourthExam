import cv2
import PIL.Image as Image

x = 19
open_path = 'D:/fourth/dataset1/{}.jpg'.format(x)
save_path = 'D:/fourth/part1/{}.png'.format(x)

# 打开图片
img = Image.open(open_path).convert('RGBA')
print(img.width, img.height)

# 缩放
scale_factor = 1.2
new_width = 800  # int(img.width * scale_factor)
new_height = 800  # int(img.height * scale_factor)
# img = img.resize((new_width, new_height), Image.NEAREST)
img = img.resize((new_width, new_height), Image.BILINEAR)
print(img.width, img.height)

# 旋转图像(center)
# img = img.rotate(180)

new_img = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))  # 透明背景
white_pixel = (255, 255, 255, 255)  # 白色

# 设置透明背景
for h in range(new_width):
    for i in range(new_width):
        if img.getpixel((h, i)) == white_pixel:
            img.putpixel((h, i), (0, 0, 0, 0))

# 偏移
horizontal_offset = 0
vertical_offset = 50
new_img.paste(img, (horizontal_offset, horizontal_offset))

# 旋转图像(origin)
new_img = new_img.rotate(10)

# 填充
for h in range(new_width):
    for i in range(new_width):
        if new_img.getpixel((h, i)) == (0, 0, 0, 0):
            new_img.putpixel((h, i), white_pixel)

new_img.save(save_path)




