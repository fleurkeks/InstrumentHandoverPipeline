import cv2
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camRecord(self.previewName, self.camID)

def camRecord(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(previewName+'.avi', fourcc, 30.0, (3840, 2160))
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
        print(frame.shape) 
    else:
        rval = False

    while rval:
        half = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
        cv2.imshow(previewName, half)
        out.write(frame)
        rval, frame = cam.read()
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("CameraRight4", 1)
thread2 = camThread("CameraLeft4", 2)
thread1.start()
thread2.start()