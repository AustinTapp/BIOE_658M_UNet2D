import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Data import *

if __name__ == "__main__":
    image = ImageData()
    print(ImageData.get_sample(image))
    for item in image:
        print(item)
