import cv2
from developement import pipeline

#video
cap = cv2.VideoCapture(r"G:\SelfDrivingcarsPractice\LaneFinding\data\videos\solidYellowLeft.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('outpy.mp4',fourcc,20, (frame_width,frame_height))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = pipeline(frame)
        cv2.imshow('Frame',frame)
        out.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
