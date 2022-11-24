#!/usr/bin/python

import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
import pickle
import pandas as pd
from math import acos, degrees
import Exercises.UpcSystemCost as UpcSystemCost
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose    

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 


def start(start,counter,sets, reps, secs, df_trainer_coords, df_trainers_costs):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sets_counter = 0 
    stframe = st.empty()
    
    resultados_acum = []
    df_results_coords_total = pd.DataFrame()
    up = False
    down = False

    while sets_counter < sets:
        # Squats reps_counter variables
        reps_counter = 0
        stage = ""
         # Load Model.

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            cap.isOpened()
            while reps_counter < reps:
                ret, frame = cap.read()
                if ret == False:
                    break
                height, width, _ = frame.shape
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                    # Concate rows
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = LoadModel().predict(X)[0]
                    body_language_prob = LoadModel().predict_proba(X)[0]
                    body_language_prob1 = body_language_prob*100
                    body_language_prob1=round(body_language_prob1[np.argmax(body_language_prob1)],2)

                    # 24. Right_hip
                    x1 = int(landmarks[24].x * width)  
                    y1 = int(landmarks[24].y * height)
                    # 26. Right_knee
                    x2 = int(landmarks[26].x * width)
                    y2 = int(landmarks[26].y * height)
                    # 28. Right_ankle
                    x3 = int(landmarks[28].x * width)
                    y3 = int(landmarks[28].y * height)
                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x3, y3])
                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)
                        

                    # Calculate angle
                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    print(f'angle: {angle}')

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )

                    df_results_coords_total = UpcSystemCost.process(image,mp_drawing,mp_pose,results,
                                                                    counter,start,df_trainer_coords,
                                                                    df_trainers_costs,df_results_coords_total,
                                                                    sets_counter,reps_counter)

                    #COLOCAR UNA MARCA DE TIEMPO
                    df_results_coords_total = df_results_coords_total.drop_duplicates(subset=0, keep="last")

                    if body_language_prob1 > 60:   
                        #SUMAR Y RESTAR UN RANGO DE 5 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                        if angle >= 160:
                                    up = True
                                    stage = "up"
                        if up == True and down == False and angle <= 70:
                                    down = True
                                    stage = "down"
                        if up == True and down == True and angle >= 160:
                                    # funcion de Costos()
                                    
                                    counter +=1
                                    start +=1
                                # inicio,c,results,resultados_acum=start_cost(inicio,c,results,resultados_acum)
                                    print(f'Paso')
                                    reps_counter += 1
                                    up = False
                                    down = False
                                    stage = "up"
                    else:
            
                        stage = ""
                        df_results_coords_total = df_results_coords_total[:-1]

                    # Setup status box
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    
                    # Set data
                    cv2.putText(image, 'SET', (15,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(sets_counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                    # Rep data
                    cv2.putText(image, 'REPS', (65,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(reps_counter), 
                                (60,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    cv2.putText(image, 'STAGE', (115,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (110,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                    
                    # Setup status box
                    # cv2.rectangle(image, (0, 480), (225, 407), (245,117,16), -1)

                    # Class data
                    cv2.putText(image, 'CLASS', (15,427), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(body_language_class), 
                                (10,467), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

                    # Prob data
                    cv2.putText(image, 'PROB', (125,427), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(body_language_prob1), 
                                (120,467), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    
                    #cv2.imshow('Mediapipe Feed', image)
                    aux_image = np.zeros(frame.shape, np.uint8)
                    cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
                    cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 0), 5)
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    cv2.fillPoly(aux_image, pts=[contours], color=(128, 0, 250))
                    image = cv2.addWeighted(image, 1, aux_image, 0.8, 0)
                    stframe.image(image,channels = 'BGR',use_column_width=True)   

                    # Used to end early
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                except:
                    pass   
            sets_counter += 1  
                       
            if (sets_counter!=sets):
                try:
                    cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    cv2.waitKey(1)
                    time.sleep(secs)   

                except:
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    pass 
                           
    cv2.rectangle(image, (50,180), (600,300), (0,255,0), -1)
    cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    #cv2.putText(image, 'REST FOR 30s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    
    df_results_coords_total.columns=["nose_x","nose_y","nose_z","nose_visibility","left_eye_inner_x",
                                    "left_eye_inner_y","left_eye_inner_z","left_eye_inner_visibility",
                                    "left_eye_x","left_eye_y","left_eye_z","left_eye_visibility",
                                    "left_eye_outer_x","left_eye_outer_y","left_eye_outer_z",
                                    "left_eye_outer_visibility","right_eye_inner_x","right_eye_inner_y",
                                    "right_eye_inner_z","right_eye_inner_visibility","right_eye_x",
                                    "right_eye_y","right_eye_z","right_eye_visibility","right_eye_outer_x",
                                    "right_eye_outer_y","right_eye_outer_z","right_eye_outer_visibility",
                                    "left_ear_x","left_ear_y","left_ear_z","left_ear_visibility",
                                    "right_ear_x","right_ear_y","right_ear_z","right_ear_visibility",
                                    "mouth_left_x","mouth_left_y","mouth_left_z","mouth_left_visibility",
                                    "mouth_right_x","mouth_right_y","mouth_right_z","mouth_right_visibility",
                                    "left_shoulder_x","left_shoulder_y","left_shoulder_z",
                                    "left_shoulder_visibility","right_shoulder_x","right_shoulder_y",
                                    "right_shoulder_z","right_shoulder_visibility","left_elbow_x",
                                    "left_elbow_y","left_elbow_z","left_elbow_visibility","right_elbow_x",
                                    "right_elbow_y","right_elbow_z","right_elbow_visibility","left_wrist_x",
                                    "left_wrist_y","left_wrist_z","left_wrist_visibility","right_wrist_x",
                                    "right_wrist_y","right_wrist_z","right_wrist_visibility","left_pinky_x",
                                    "left_pinky_y","left_pinky_z","left_pinky_visibility","right_pinky_x",
                                    "right_pinky_y","right_pinky_z","right_pinky_visibility","left_index_x",
                                    "left_index_y","left_index_z","left_index_visibility","right_index_x",
                                    "right_index_y","right_index_z","right_index_visibility","left_thumb_x",
                                    "left_thumb_y","left_thumb_z","left_thumb_visibility","right_thumb_x",
                                    "right_thumb_y","right_thumb_z","right_thumb_visibility","left_hip_x",
                                    "left_hip_y","left_hip_z","left_hip_visibility","right_hip_x",
                                    "right_hip_y","right_hip_z","right_hip_visibility","left_knee_x",
                                    "left_knee_y","left_knee_z","left_knee_visibility","right_knee_x",
                                    "right_knee_y","right_knee_z","right_knee_visibility","left_ankle_x",
                                    "left_ankle_y","left_ankle_z","left_ankle_visibility","right_ankle_x",
                                    "right_ankle_y","right_ankle_z","right_ankle_visibility","left_heel_x",
                                    "left_heel_y","left_heel_z","left_heel_visibility","right_heel_x",
                                    "right_heel_y","right_heel_z","right_heel_visibility",
                                    "left_foot_index_x","left_foot_index_y","left_foot_index_z",
                                    "left_foot_index_visibility","right_foot_index_x","right_foot_index_y",
                                    "right_foot_index_z","right_foot_index_visibility","starting_cost",
                                    "final_cost","resulting_cost","message_validation","sets_counter",
                                    "reps_counter","pose"]

    first_column = df_results_coords_total.pop('pose')    
    now = datetime.now()    
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    df_results_coords_total.insert(0, 'pose', first_column)
    
    df_results_coords_total = df_results_coords_total.drop_duplicates(subset='pose', keep="last")
    df_results_coords_total.to_csv("./resultados_costos/Squats_resultados_costos_"+str(date_time)+".csv",index=False)   
    #cv2.imshow('Mediapipe Feed', image)
    stframe.image(image,channels = 'BGR',use_column_width=True)
    #cv2.waitKey(1) 
    time.sleep(10)          
    cap.release()
    cv2.destroyAllWindows()
    #cv2.destroyAllWindows()

def LoadModel():
    model_weights = './Exercises/model_weights/weights_body_language.pkl'
    with open(model_weights, "rb") as f:
        model = pickle.load(f)
    return model