import cv2
import mediapipe as mp
import time

#inbuilt library for drawing over image
mp_drawing = mp.solutions.drawing_utils                 
mp_drawing_styles = mp.solutions.drawing_styles
 
# initializing face_mesh from mediapipe
mp_face_mesh = mp.solutions.face_mesh

# changing the 2 paramteres will change the points and lines drawn over image 
# by default lines 64 to 89 are commented which uses these functions you can uncommet them and see the difference urself
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture('test.mp4')          #for a video file input
# cap = cv2.VideoCapture(0)                 #for webcam input

# reading height and width of your video/webcam input 
success, image = cap.read()
width = image.shape[1]
height = image.shape[0]

res = []


fourcc = cv2.VideoWriter_fourcc(*"XVID")

# create the video write object
out = cv2.VideoWriter("output.avi", fourcc, 30, (width, height))


# font type to write over image   
font = cv2.FONT_HERSHEY_PLAIN

# frames counter and start time to determine frame rate
frames = 0
start_time = time.time()

# initializing with desired parameters(names are self explanatory)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    try:
        while cap.isOpened():
            frames += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            
            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                   
                    # each block of below codes have their own drawing abilities uncomment them one by one to check them out
                    
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_tesselation_style())
                   
                    
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_contours_style())
                   
                    
                   # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     # connections=mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     # connection_drawing_spec=mp_drawing_styles
                    #     # .get_default_face_mesh_iris_connections_style())
                    #     )
                    
                    
                    # loop to draw circles at landmarks
                    for idd, lm in  enumerate(face_landmarks.landmark):
                        image = cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 2, (255, 255, 0), 1)
                    
                        
                    
            # writing FPS on each frame
            cv2.putText(image, 'FPS: ' + str(round(frames /(time.time() - start_time), 2)) , (10 , 40), font, fontScale = 2, color=(255, 55, 55), thickness=2)
            
            # writing frame to output video
            out.write(image)
            
            # showing frame on window
            cv2.imshow('Result Face Mesh', image)
            
            # on pressing esc key loop/video reading stops
            key =  cv2.waitKey(1) 
            if key == 27:
              break
            
        # when video ends or esc key is presses everything stops without loss
        cap.release()
        out.release()
        cv2.destroyAllWindows()
       
    # if something unexpected happens everything stops without loss
    except:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    

    
