from PIL import Image
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np
from util import crop
import random

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_prediction(img_path, threshold):
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)[:3,:,:]
  pred = model([img])
  masks = [
    mask
    for i, mask in enumerate((pred[0]['masks']>0.01).squeeze().detach().cpu().numpy())
    if pred[0]['scores'][i].detach().numpy() > threshold and
      pred[0]['labels'][i].numpy() == 1
  ]
  return masks

def get_masks(image):
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = [255, 255, 255]
  masks = np.stack([r, g, b], axis=2)
  return masks

def instance_segmentation_api(img_path, threshold=0.5):
  masks = get_prediction(img_path, threshold)
  img = cv2.imread(img_path)
  mask1 = None
  mask2 = None
  mask_all = None
  persons1 = None
  persons2 = None
  for i in range(len(masks)):
    mask_ = get_masks(masks[i])
    if (mask1 is None):
      mask1 = mask_
    elif (mask2 is None):
      mask2 = mask_
    else:
      if (random.random() < 0.5):
        mask1 = cv2.addWeighted(mask1, 1, mask_, 1, 0)
      else:
        mask2 = cv2.addWeighted(mask2, 1, mask_, 1, 0)
  if (mask1 is not None and mask2 is not None):
    mask_all = cv2.addWeighted(mask1, 1, mask2, 1, 0)
  elif (mask2 is None):
    mask_all = mask1
  if (mask_all is None):
    output = img
  else:
    output = cv2.addWeighted(img, 1, mask_all, 1, 0)
  if (mask1 is not None):
    persons1 = cv2.bitwise_and(img, mask1)
  if (mask2 is not None):
    persons2 = cv2.bitwise_and(img, mask2)
  output = crop(output)
  if (mask_all is not None):
    mask_all = crop(mask_all)
  if (persons1 is not None):
    persons1 = crop(persons1)
  if (persons2 is not None):
    persons2 = crop(persons2)
  # cv2.imwrite('input.png', output)
  # cv2.imwrite('mask.png', get_transparent_img(mask))
  # cv2.imwrite('persons.png', get_transparent_img(persons))
  return output, mask_all, persons1, persons2
