import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import stag
from rembg import remove
stag_id = 8
med_type = "Pill"
image_path = ".\\thiago_fotos_MED\\img_8_008.jpg"
neg_image_1 = ".\\thiago_fotos_SUB\\img_2_009.jpg"
neg_image_2 = ".\\thiago_fotos_SUB\\img_8_007.jpg"
image_a = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
image_b = cv2.imread(neg_image_1, cv2.COLOR_BGR2RGB)
image_c = cv2.imread(neg_image_2, cv2.COLOR_BGR2RGB)
subtracted_image = cv2.subtract(image_a, image_b )
subtracted_image_2 = cv2.subtract(subtracted_image, image_c )
plt.imshow(cv2.cvtColor(subtracted_image, cv2.COLOR_BGR2RGB))
plt.title('subtracted')
plt.show()
plt.imshow(cv2.cvtColor(subtracted_image_2, cv2.COLOR_BGR2RGB))
plt.title('subtracted')
plt.show()
