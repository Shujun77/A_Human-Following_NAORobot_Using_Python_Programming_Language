import urllib.request
import cv2
import os
import numpy as np


if not os.path.exists('neg'):
    os.makedirs('neg')

neg_img_url = ['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02123597',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03223299',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00017222',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09416076',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03563967',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03094503',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03100490',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07557434',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n06470073'
               ]

urls = ''
for img_url in neg_img_url:
    urls += urllib.request.urlopen(img_url).read().decode()

img_index = 1
for url in urls.split('\n'):
    try:
        print(url)
        urllib.request.urlretrieve(url, 'neg/' + str(img_index) + '.jpg')
        gray_img = cv2.imread('neg/' + str(img_index) + '.jpg', cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(gray_img, (150, 150))
        cv2.imwrite('neg/' + str(img_index) + '.jpg', image)
        img_index += 1
    except Exception as e:
        print(e)


def is_same_image(img_file1, img_file2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    if img1.shape == img2.shape and not (np.bitwise_xor(img1, img2).any()):
        return True
    else:
        return False



