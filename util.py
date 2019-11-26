import cv2

def crop(image):
  grid = 8
  h, w, _ = image.shape
  return image[:h//grid*grid, :w//grid*grid, :]

def get_transparent_img(img):
  tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
  b, g, r = cv2.split(img)
  rgba = [b,g,r, alpha]
  return cv2.merge(rgba,4)