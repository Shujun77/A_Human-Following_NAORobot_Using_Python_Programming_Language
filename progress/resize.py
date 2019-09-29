import cv2

img_index = 1
try:
    img = cv2.imread('/Users/shujunliu/Desktop/progress/box.jpg', cv2.IMREAD_UNCHANGED)
    gray_img = cv2.imread('/Users/shujunliu/Desktop/progress/box.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(gray_img, (50, 30))
    cv2.imwrite(str(img_index) + '.jpg', image)
except Exception as e:
    print(e)
