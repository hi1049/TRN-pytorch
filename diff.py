import PIL
from PIL import ImageChops
from PIL import Image


im1 = Image.open('video_datasets/jester/20bn-jester-v1/1/00001.jpg')
im2 = Image.open('video_datasets/jester/20bn-jester-v1/1/00037.jpg')
diff = ImageChops.difference(im1, im2)
diff.save('test.jpg')
diff.show('diff')

