import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils                 #inbuilt library for drawing over image
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture('test.mp4')          #for a video file input
# cap = cv2.VideoCapture(0)                 #for webcam input

success, image = cap.read()
width = image.shape[1]
height = image.shape[0]

res = []

fourcc = cv2.VideoWriter_fourcc(*"XVID")

# create the video write object
out = cv2.VideoWriter("output.avi", fourcc, 30, (width, height))


# res[0].landmark.__hash__     
font = cv2.FONT_HERSHEY_PLAIN
frames = 0
start_time = time.time()
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    
    # refine_landmarks=True,
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
                    
                    res.append(face_landmarks)
                    
                    
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
                    
                    for idd, lm in  enumerate(face_landmarks.landmark):
                        image = cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 2, (255, 255, 0), 1)
                    
                        
                    
                  
            cv2.putText(image, 'FPS: ' + str(round(frames /(time.time() - start_time), 2)) , (10 , 40), font, fontScale = 2, color=(255, 55, 55), thickness=2)
            out.write(image)
            cv2.imshow('Result Face Mesh', image)
            key =  cv2.waitKey(1) 
            if key == 27:
              break
          
    except:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    

    