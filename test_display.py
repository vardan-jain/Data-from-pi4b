import time
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import Image

# NB ssd1306 devices are monochromatic; a pixel is enabled with
#    white and disabled with black.
# NB the ssd1306 class has no way of knowing the device resolution/size.
device = ssd1306(i2c(port=1, address=0x3c), width=128, height=64, rotate=0)

# set the contrast to minimum.
device.contrast(1)

print("hello")

# NB this will only send the data to the display after this "with" block is complete.
# NB the draw variable is-a PIL.ImageDraw.Draw (https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html).
# see https://github.com/rm-hull/luma.core/blob/master/luma/core/render.py
with canvas(device, dither=True) as draw:
    #draw.rectangle(device.bounding_box, outline='white', fill='black')
    
    message = 'Shivanshu'
    text_size = draw.textsize(message)
    draw.text((device.width - text_size[0], (device.height - text_size[1]) // 2), message, fill='white')

# NB the display will be turn off after we exit this application.
time.sleep(5*60)
