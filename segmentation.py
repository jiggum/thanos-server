from PIL import Image
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np
from util import crop

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_prediction(img_path, threshold):
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  masks = [
    mask
    for i, mask in enumerate((pred[0]['masks']>0.05).squeeze().detach().cpu().numpy())
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
  mask = None
  for i in range(len(masks)):
    mask_ = get_masks(masks[i])
    if (i == 0):
      mask = mask_
    mask = cv2.addWeighted(mask, 1, mask_, 1, 0)

  output = cv2.addWeighted(img, 1, mask, 1, 0)
  print(img.shape, mask.shape)
  persons = cv2.bitwise_and(img, mask)

  output = crop(output)
  mask = crop(mask)
  persons = crop(persons)
  # cv2.imwrite('input.png', output)
  # cv2.imwrite('mask.png', get_transparent_img(mask))
  # cv2.imwrite('persons.png', get_transparent_img(persons))
  return output, mask, persons
