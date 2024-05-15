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
    cam = cv2.VideoCapture('http://admin:admin1234@192.168.50.62/2')
    cam.set(cv2.CAP_PROP_FPS, 50)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(previewName+'.avi', fourcc, 30.0, (1280, 720))
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
        print(frame.shape) 
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        out.write(frame)
        rval, frame = cam.read()
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("CameraLeft5", 1)
#thread2 = camThread("CameraRight5", 2)
thread1.start()
#thread2.start()