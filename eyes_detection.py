import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

p_left_eye = [385, 380, 387, 373, 362, 263]
p_right_eye = [160, 144, 158, 153, 33, 133]

p_eyes = p_left_eye+p_right_eye
p_eyes

def calculus_ear(face, p_right_eye,p_left_eye):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        left_face = face[p_left_eye,:]
        right_face = face[p_right_eye,:]

        left_ear = (np.linalg.norm(left_face[0]-left_face[1])+np.linalg.norm(left_face[2]-left_face[3]))/(2*(np.linalg.norm(left_face[4]-left_face[5])))
        right_ear = (np.linalg.norm(right_face[0]-right_face[1])+np.linalg.norm(right_face[2]-right_face[3]))/(2*(np.linalg.norm(right_face[4]-right_face[5])))
    except:
        left_ear = 0.0
        right_ear = 0.0
    media_ear = (left_ear+right_ear)/2
    return media_ear

p_mouth = [82, 87, 13, 14, 312, 317, 78, 308]

def calculus_mar(face,p_mouth):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        mouth_face = face[p_mouth,:]

        mar = (np.linalg.norm(mouth_face[0]-mouth_face[1])+np.linalg.norm(mouth_face[2]-mouth_face[3])+np.linalg.norm(mouth_face[4]-mouth_face[5]))/(2*(np.linalg.norm(mouth_face[6]-mouth_face[7])))
    except:
        mar = 0.0

    return mar

ear_limiar = 0.3
mar_limiar = 0.1
sleeping = 0
blink_count = 0
c_time = 0
temporary_count = 0
list_count = []

t_blink = time.time()

cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        suc, frame = cap.read()
        if not success:
            print('Ignoring the empty camera frame.')
            continue
        length, width, _ = frame.shape

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        exit_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            for face_landmarks in exit_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255,102,102),thickness=1,circle_radius=1),
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(102,204,0),thickness=1,circle_radius=1))
                face = face_landmarks.landmark
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in p_eyes:
                       coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y, width, length)
                       cv2.circle(frame, coord_cv, 2, (255,0,0), -1)
                    if id_coord in p_mouth:
                       coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y, width, length)
                       cv2.circle(frame, coord_cv, 2, (255,0,0), -1)

                ear = calculus_ear(face,p_right_eye, p_left_eye)
                cv2.rectangle(frame, (0,1),(290,140),(58,58,55),-1)
                cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.9, (255, 255, 255), 2)
                mar = calculus_mar(face,p_mouth)
                cv2.putText(frame, f"MAR: {round(mar, 2)} {'Open' if mar>=mar_limiar else 'Closed'}", (1, 50),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.9, (255, 255, 255), 2)

                if ear < ear_limiar and mar < mar_limiar:
                    t_initial = time.time() if sleeping == 0 else t_initial
                    blink_count = blink_count+1 if sleeping == 0 else blink_count
                    sleeping = 1
                if (sleeping == 1 and ear >= ear_limiar) or (ear <= ear_limiar and mar>= mar_limiar):
                    sleeping = 0
                t_end = time.time()
                elapsed_time = t_end - t_blink

                if elapsed_time >= (c_time+1):
                    c_tempo = elapsed_time
                    blink_ps = blink_count-temporary_count
                    temporary_count = blink_count
                    list_count.append(blink_ps)
                    list_count = list_count if (len(list_count)<=60) else list_count[-60:]
                blink_pm = 15 if elapsed_time<=60 else sum(list_count)


                cv2.putText(frame, f"Blink: {blink_count}", (1, 120),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.9, (109, 233, 219), 2)
                time = (t_end-t_initial) if sleeping == 1 else 0.0
                cv2.putText(frame, f"Time: {round(time, 3)}", (1, 80),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.9, (255, 255, 255), 2)

                if blink_pm < 10 or time>=1.5:
                    cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
                    cv2.putText(frame, f"You might be sleepy,", (60, 420),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.85, (58,58,55), 1)
                    cv2.putText(frame, f"consider resting.", (180, 450),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.85, (58,58,55), 1)

        except:
            pass

        cv2.imshow('Camera',frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break
cap.release()
cv2.destroyAllWindows()
