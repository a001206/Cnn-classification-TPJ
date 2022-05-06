# import cv2

# cap = cv2.VideoCapture(0)

# cap.set(3, 320)
# cap.set(4,240)

# while True:
#     ret, frame = cap.read()

#     if ret:
#         cv2.imshow('video', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
import cv2

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 180)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

cv2.namedWindow("Haesung Webcam!!")

img_counter = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Haesung Webcam!!", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "img{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

# img = cv2.imread('1.jpg', 1)
# path = 'D:/OpenCV/Scripts/Image s'
# cv2.imwrite(os.path.join(path , 'waka.jpg'), img)
# cv2.waitKey(0)


'/Users/ganghaeseong/Documents/tf/Cnn-classification-TPJ/cam'
cam.release()

cv2.destroyAllWindows() 