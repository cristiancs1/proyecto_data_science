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

def calculate_angleacos(a,b,c):
    angle = degrees(acos((a**2+c**2-b**2) / (2 * a * c)))
    angle = int(angle)
    return angle 

def get_angle(df, index, part):
    angle_in=df['Angulo'][(df.pose==index+1)&(df.Parte==part)]
    angle_in=angle_in.iloc[0]
    return angle_in

def get_desv_angle(df, index, part):
    desv_in=df['Desviacion_estandar'][(df.pose==index+1)&(df.Parte==part)]
    desv_in=desv_in.iloc[0]
    return desv_in

def start(start,counter,sets, reps, secs, df_trainer_coords, df_trainers_costs,df_trainers_angles):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sets_counter = 0 
    stframe = st.empty()
    
    resultados_acum = []
    df_results_coords_total = pd.DataFrame()
    up = False
    down = False
    start = 0
    while sets_counter < sets:
        # Squats reps_counter variables
        reps_counter = 0
        stage = ""

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            cap.isOpened()
            while reps_counter < reps:
                ret, frame = cap.read()
                if ret == False:
                    break
                frame = cv2.flip(frame,1)
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
                # try:
                if results.pose_landmarks is None:
                            cv2.putText(image, 
                            "No se han detectado ninguno de los 33 puntos corporales",
                            (100,250),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            (0, 0, 255),
                            1, 
                            cv2.LINE_AA)
                            stframe.image(image,channels = 'BGR',use_column_width=True)   

                else:
                    landmarks = results.pose_landmarks.landmark

                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                    # Concate rows
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])

                    # Load Model Clasification
                    body_language_class = LoadModel().predict(X)[0]
                    body_language_prob = LoadModel().predict_proba(X)[0]
                    body_language_prob1 = body_language_prob*100
                    body_language_prob1=round(body_language_prob1[np.argmax(body_language_prob1)],2)

                    right_arm_x1 = int(landmarks[12].x * width) #right_arm
                    right_arm_x2 = int(landmarks[14].x * width)
                    right_arm_x3 = int(landmarks[16].x * width)
                    right_arm_y1 = int(landmarks[12].y * height)
                    right_arm_y2 = int(landmarks[14].y * height)
                    right_arm_y3 = int(landmarks[16].y * height)  

                    right_arm_p1 = np.array([right_arm_x1, right_arm_y1])
                    right_arm_p2 = np.array([right_arm_x2, right_arm_y2])
                    right_arm_p3 = np.array([right_arm_x3, right_arm_y3])

                    right_arm_l1 = np.linalg.norm(right_arm_p2 - right_arm_p3)
                    right_arm_l2 = np.linalg.norm(right_arm_p1 - right_arm_p3)
                    right_arm_l3 = np.linalg.norm(right_arm_p1 - right_arm_p2)

                    # Calculate angle
                    right_arm_angle = calculate_angleacos(right_arm_l1, right_arm_l2, right_arm_l3)
                    print(f'right_arm_angle: {right_arm_angle}')

                    right_torso_x1 = int(landmarks[12].x * width) #right_torso
                    right_torso_x2 = int(landmarks[24].x * width)
                    right_torso_x3 = int(landmarks[26].x * width) 
                    right_torso_y1 = int(landmarks[12].y * height)
                    right_torso_y2 = int(landmarks[24].y * height)
                    right_torso_y3 = int(landmarks[26].y * height) 

                    right_torso_p1 = np.array([right_torso_x1, right_torso_y1])
                    right_torso_p2 = np.array([right_torso_x2, right_torso_y2])
                    right_torso_p3 = np.array([right_torso_x3, right_torso_y3])

                    right_torso_l1 = np.linalg.norm(right_torso_p2 - right_torso_p3)
                    right_torso_l2 = np.linalg.norm(right_torso_p1 - right_torso_p3)
                    right_torso_l3 = np.linalg.norm(right_torso_p1 - right_torso_p2)

                    # Calculate angle
                    right_torso_angle = calculate_angleacos(right_torso_l1, right_torso_l2, right_torso_l3)
                    print(f'right_torso_angle: {right_torso_angle}')

                    right_leg_x1 = int(landmarks[24].x * width) #right_leg
                    right_leg_x2 = int(landmarks[26].x * width)
                    right_leg_x3 = int(landmarks[28].x * width) 
                    right_leg_y1 = int(landmarks[24].y * height)
                    right_leg_y2 = int(landmarks[26].y * height)
                    right_leg_y3 = int(landmarks[28].y * height)

                    right_leg_p1 = np.array([right_leg_x1, right_leg_y1])
                    right_leg_p2 = np.array([right_leg_x2, right_leg_y2])
                    right_leg_p3 = np.array([right_leg_x3, right_leg_y3])

                    right_leg_l1 = np.linalg.norm(right_leg_p2 - right_leg_p3)
                    right_leg_l2 = np.linalg.norm(right_leg_p1 - right_leg_p3)
                    right_leg_l3 = np.linalg.norm(right_leg_p1 - right_leg_p2)

                    # Calculate angle
                    right_leg_angle = calculate_angleacos(right_leg_l1, right_leg_l2, right_leg_l3)
                    print(f'right_leg_angle: {right_leg_angle}')

                    df_results_coords_total = UpcSystemCost.process(image,mp_drawing,mp_pose,results,
                                                                    counter,start,df_trainer_coords,
                                                                    df_trainers_costs,df_results_coords_total,
                                                                    sets_counter,reps_counter)

                    #COLOCAR UNA MARCA DE TIEMPO
                    df_results_coords_total = df_results_coords_total.drop_duplicates(subset=0, keep="last")

                    if body_language_prob1 > 51: 

                        print(f'body_language_prob1: {body_language_prob1}')
                        print(f'start: {start}')
                        right_arm_angle_in= get_angle(df_trainers_angles, start, 'right_arm_angles')
                        print(f'right_arm_angle_in: {right_arm_angle_in}')
                        right_torso_angle_in=get_angle(df_trainers_angles, start, 'right_torso_angles')
                        print(f'right_torso_angle_in: {right_torso_angle_in}')
                        right_leg_angle_in=get_angle(df_trainers_angles, start, 'right_leg_angles')
                        print(f'right_leg_angle_in: {right_leg_angle_in}')
                        desv_right_arm_angle_in=get_desv_angle(df_trainers_angles, start, 'right_arm_angles')
                        print(f'desv_right_arm_angle: {desv_right_arm_angle_in}')
                        desv_right_torso_angle_in=get_desv_angle(df_trainers_angles, start, 'right_torso_angles')
                        print(f'desv_right_torso_angle: {desv_right_torso_angle_in}')
                        desv_right_leg_angle_in=get_desv_angle(df_trainers_angles, start, 'right_leg_angles')
                        print(f'desv_right_leg_angle: {desv_right_leg_angle_in}')

                        #SUMAR Y RESTAR UN RANGO DE 30 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                        if  up == False and right_arm_angle in range(int(right_arm_angle_in-desv_right_arm_angle_in), int(right_arm_angle_in + desv_right_arm_angle_in + 1)) and right_torso_angle in range(int(right_torso_angle_in - desv_right_torso_angle_in), int(right_torso_angle_in + desv_right_torso_angle_in + 1)) and right_leg_angle in range(int(right_leg_angle_in - desv_right_leg_angle_in),int(right_leg_angle_in + desv_right_leg_angle_in + 1)):
                            up = True
                            stage = "up"
                            start +=1
                            print(f'Paso Primera Pose')
                        elif up == True and down == False and right_arm_angle in range(int(right_arm_angle_in - desv_right_arm_angle_in) , int(right_arm_angle_in + desv_right_arm_angle_in + 1)) and right_torso_angle in range(int(right_torso_angle_in - desv_right_torso_angle_in), int(right_torso_angle_in + desv_right_torso_angle_in + 1)) and right_leg_angle in range(int(right_leg_angle_in - desv_right_leg_angle_in),int(right_leg_angle_in + desv_right_leg_angle_in + 1)):
                            down = True
                            stage = "down"
                            start +=1
                            print(f'Paso Segunda Pose')
                        elif up == True and down == True and right_arm_angle in range(int(right_arm_angle_in - desv_right_arm_angle_in) , int(right_arm_angle_in + desv_right_arm_angle_in + 1)) and right_torso_angle in range(int(right_torso_angle_in - desv_right_torso_angle_in), int(right_torso_angle_in + desv_right_torso_angle_in + 1)) and right_leg_angle in range(int(right_leg_angle_in - desv_right_leg_angle_in),int(right_leg_angle_in + desv_right_leg_angle_in + 1)):
                            # funcion de Costos()                       
                            counter +=1
                            print(f'Paso Tercera Pose')
                            reps_counter += 1
                            up = False
                            down = False
                            stage = "up"
                            start = 0
                        else:
                            df_results_coords_total = df_results_coords_total[:-1]
                    else:
            
                        stage = ""
                        start = 0
                        up = False
                        print(f'Salio')
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
                    

                    cv2.line(image, (right_arm_x1, right_arm_y1), (right_arm_x2, right_arm_y2), (242, 14, 14), 3)
                    cv2.line(image, (right_arm_x2, right_arm_y2), (right_arm_x3, right_arm_y3), (242, 14, 14), 3)
                    cv2.circle(image, (right_arm_x1, right_arm_y1), 6, (128, 0, 255),-1)
                    cv2.circle(image, (right_arm_x2, right_arm_y2), 6, (128, 0, 255),-1)
                    cv2.circle(image, (right_arm_x3, right_arm_y3), 6, (128, 0, 255),-1)
                    cv2.putText(image, str(int(right_arm_angle)), (right_arm_x2 + 30, right_arm_y2), 1, 1.5, (128, 0, 250), 2)

                    cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                    cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                    cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                    cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                    cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                    cv2.putText(image, str(int(right_torso_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                    cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                    cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                    cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                    cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                    cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                    cv2.putText(image, str(int(right_leg_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                    stframe.image(image,channels = 'BGR',use_column_width=True)   
                    key = cv2.waitKey(0) 
                    leter = ord('q')
                    print(f'Key: {key}')
                    print(f'leter: {leter}')
                    # Used to end early
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # except:
                #     pass   
            sets_counter += 1  
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
            if (sets_counter!=sets):
                try:
                    cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                    cv2.putText(image, 'REST FOR ' + str(secs) + ' s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    # cv2.waitKey(1)
                    time.sleep(secs)   

                except:
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    pass 
                           
    cv2.rectangle(image, (50,180), (600,300), (0,255,0), -1)
    cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    
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
    #Faltaría: clase, probabilidad, right arm, right torso, right leg
    df_results_coords_total.to_csv("./resultados_costos/Squats_resultados_costos_"+str(date_time)+".csv",index=False)   
    stframe.image(image,channels = 'BGR',use_column_width=True)
    time.sleep(5)          
    cap.release()
    cv2.destroyAllWindows()

def LoadModel():
    model_weights = './Exercises/model_weights/weights_body_language.pkl'
    with open(model_weights, "rb") as f:
        model = pickle.load(f)
    return model