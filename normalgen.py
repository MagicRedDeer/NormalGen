from PIL import Image
import numpy as np
import math
import os


image_path = 'c:/Users/Quixel/Downloads/stones-right.png'

image_dir, image_name = os.path.split(image_path)
image_name, image_ext = os.path.splitext(image_name)


im = Image.open(image_path)
gray = im.convert('L')

new_img = Image.new('RGB', im.size, (0,255,0))


print ( im.size )
for x in range(im.size[0]):
    for y in range(im.size[1]):
        if x < im.size[0]-1 and x > 0 and y < im.size[1]-1 and y > 0:
            dx = gray.getpixel((x+1, y)) - gray.getpixel((x-1, y))
            dy = gray.getpixel((x, y+1)) - gray.getpixel((x, y-1))
            k = np.array((
                (dx+255)/2,
                (dy+255)/2,
                255))
            k = k / np.linalg.norm(k)
            k *= 255
            new_img.putpixel(
                    (x, y), (int(k[0]), int(k[1]), int(k[2])))

gray.show()
new_img.show()


new_img.save(os.path.join(image_dir, image_name + '_NMAP.' + image_ext))
