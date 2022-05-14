import cv2

img = cv2.imread('img0.png')
print(img.shape)
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()