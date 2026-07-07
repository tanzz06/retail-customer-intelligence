import cv2

cap = cv2.VideoCapture('fixed_video.mp4', cv2.CAP_FFMPEG)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 100 == 0:
        print(f"Read {count} frames successfully")

print(f"Total frames read: {count}")
cap.release()