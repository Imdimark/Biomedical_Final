# General imports
from turtle import bgcolor
import numpy as np
import pandas as pd
import os
import time
from math import sqrt
import rospy
import roslaunch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
# For multithreading
from threading import Thread, Lock
import queue
# For OpenCV
import cv2
# For GUI
import tkinter as tk
from tkinter import Label, Button, BooleanVar, Checkbutton, Text, ttk

# For pygame
import pygame
# For reaching task 
from my_python__common.reaching import Reaching
from my_python__common.stopwatch import StopWatch
from my_python__common.filter_butter_online import FilterButter3
import my_python__common.reaching_functions
import my_python__common.python_code
# For controlling computer cursor
import pyautogui
# For Mediapipe
import mediapipe as mp
# For training pca/autoencoder
from my_python__common.compute_bomi_map import Autoencoder, PrincipalComponentAnalysis, compute_vaf
from sklearn.manifold import TSNE
pyautogui.PAUSE = 0.01  # set fps of cursor to 100Hz ish when mouse_enabled is True

# Global variables initialisation
MASTER=None
prev_cx=0
prev_cy=0
model=''

class MainApplication(tk.Frame):
    """
    class that defines the main tkinter window --> graphic with buttons etc..
    """

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        global MASTER
        self.parent = parent
        self.calibPath = os.path.dirname(os.path.abspath(__file__)) + "/calib/"
        self.drPath = ''
        self.num_joints = 0
        self.joints = np.zeros((6, 1))
        self.dr_mode = 'ae'
        self.font_size = 18
        MASTER = self.master
       
        self.btn_num_joints = Button(parent, text="Select Joints", borderwidth=3, relief="solid", command=self.select_joints,bg='#006600',fg='black')
        self.btn_num_joints.config(font=("Arial", self.font_size))
        self.btn_num_joints.grid(row=0, column=0, columnspan=2, padx=20, pady=30, sticky='nesw')

        # set checkboxes for selecting joints
        self.check_nose = BooleanVar()
        self.check1 = Checkbutton(win, text="Nose", variable=self.check_nose,background="#6495ED")
        self.check1.config(font=("Arial", self.font_size))
        self.check1.grid(row=0, column=2, padx=(0, 40), pady=30, sticky='w')

        self.check_eyes = BooleanVar()
        self.check2 = Checkbutton(win, text="Eyes", variable=self.check_eyes,background="#6495ED")
        self.check2.config(font=("Arial", self.font_size))
        self.check2.grid(row=0, column=3, padx=(0, 40), pady=30, sticky='w')

        self.check_shoulders = BooleanVar()
        self.check3 = Checkbutton(win, text="Shoulders", variable=self.check_shoulders,background="#6495ED")
        self.check3.config(font=("Arial", self.font_size))
        self.check3.grid(row=0, column=4, padx=(0, 30), pady=30, sticky='w')

        self.check_forefinger = BooleanVar()
        self.check4 = Checkbutton(win, text="Right Forefinger",
                                  variable=self.check_forefinger,background="#6495ED")
        self.check4.config(font=("Arial", self.font_size))
        self.check4.grid(row=0, column=5, padx=(0, 20), pady=30, sticky='w')

        self.check_fingers = BooleanVar()
        self.check5 = Checkbutton(win, text="Fingers", variable=self.check_fingers,background="#6495ED")
        self.check5.config(font=("Arial", self.font_size))
        self.check5.grid(row=0, column=6, padx=(0, 20), pady=30, sticky='nesw')

        self.check_mouth = BooleanVar()
        self.check6 = Checkbutton(win, text="Mouth", variable=self.check_mouth,background="#6495ED")
        self.check6.config(font=("Arial", self.font_size))
        self.check6.grid(row=0, column=7, padx=(0, 20), pady=30, sticky='nesw')

        self.btn_calib = Button(parent, text="Calibration", command=self.calibration, borderwidth=3, relief="solid",bg='#CCCCFF',fg='black')
        self.btn_calib["state"] = "disabled"
        self.btn_calib.config(font=("Arial", self.font_size))
        self.btn_calib.grid(row=1, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')
        self.calib_duration = 3000

    

        # Calibration time remaining
        self.lbl_calib = Label(win, text='Calibration time: ')
        self.lbl_calib.config(font=("Arial", self.font_size),background="#6495ED")
        self.lbl_calib.grid(row=1, column=2, columnspan=2, pady=(20, 30), sticky='w')

        # BoMI map button and checkboxes
        self.btn_map = Button(parent, text="Calculate BoMI Map",  borderwidth=3, relief="solid", command=self.train_map, bg='#006600', fg='Black')
        self.btn_map["state"] = "disabled"
        self.btn_map.config(font=("Arial", self.font_size))
        self.btn_map.grid(row=2, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        self.check_pca = BooleanVar(value=True)
        self.check_pca1 = Checkbutton(win, text="PCA", variable=self.check_pca,background="#6495ED")
        self.check_pca1.config(font=("Arial", self.font_size))
        self.check_pca1.grid(row=2, column=2, padx=(0, 20), pady=(20, 30), sticky='w')

        self.check_ae = BooleanVar()
        self.check_ae1 = Checkbutton(win, text="AE", variable=self.check_ae,background="#6495ED")
        self.check_ae1.config(font=("Arial", self.font_size))
        self.check_ae1.grid(row=2, column=3, padx=(0, 20), pady=(20, 30), sticky='w')

        self.check_vae = BooleanVar()
        self.check_vae1 = Checkbutton(win, text="Variational AE", variable=self.check_vae,background="#6495ED")
        self.check_vae1.config(font=("Arial", self.font_size))
        self.check_vae1.grid(row=2, column=4, pady=(20, 30), sticky='w')
        
        self.check_tsne = BooleanVar()
        self.check_tsne1= Checkbutton(win, text="TSNE", variable=self.check_tsne,background="#6495ED")
        self.check_tsne1.config(font=("Arial", self.font_size))
        self.check_tsne1.grid(row=2, column=5, pady=(30, 40), sticky='w')

       
        self.btn_custom = Button(parent, text="Customization", command=self.customization, borderwidth=3, relief="solid",bg='#CCCCFF',fg='black')
        self.btn_custom["state"] = "disabled"
        self.btn_custom.config(font=("Arial", self.font_size))
        self.btn_custom.grid(row=3, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        self.btn_start = Button(parent, text="Practice",borderwidth=3, relief="solid", command=self.start, bg='#006600',fg='black')
        self.btn_start["state"] = "disabled"
        self.btn_start.config(font=("Arial", self.font_size))
        self.btn_start.grid(row=4, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        # set label for number of target remaining
        self.lbl_tgt = Label(win, text='Remaining targets: ')
        self.lbl_tgt.config(font=("Arial", self.font_size),background="#6495ED")
        self.lbl_tgt.grid(row=4, column=3, pady=(20, 30), columnspan=2, sticky='w')

        self.check_mouse = BooleanVar()
        self.check6 = Checkbutton(win, text="Real Mouse", variable=self.check_mouse,background="#6495ED")
        self.check6.config(font=("Arial", self.font_size))
        self.check6.grid(row=4, column=2, padx=(0, 20), pady=(20, 30), sticky='nesw')
        

        #############################################################
        #ROS button
        self.btn_ros = Button(parent, text="ROS game", command=self.ROS_game, borderwidth=3, relief="solid",bg='#CCCCFF',fg='black')
        self.btn_ros.config(font=("Arial", self.font_size))
        self.btn_ros.grid(row=8, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')
	#############################################################
        self.btn_close = Button(parent, text="Close", command=parent.destroy, bg="#800000", borderwidth=3, relief="solid")
        self.btn_close.config(font=("Arial", self.font_size))
        self.btn_close.grid(row=9, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')
	
    def ROS_game(self):
        '''uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)
	launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/haier/catkin_ws/src/testapi/launch/test_node.launch"])
	launch.start()
    	rospy.sleep(3)
	# 3 seconds later
	launch.shutdown()'''
        package = 'stage_ros'
        executable = 'stageros'
        #node = roslaunch.core.Node(package, executable, args='$(rospack find stage_ros)/world/willow-erratic.world')
        node = roslaunch.core.Node(package, executable, args='/root/ros_ws/src/stage_ros/world/erratic-inc.world')

        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()

        process = launch.launch(node)
        print (process.is_alive())
        #process.stop()
        
        #apro finestra con i 4 punti
        self.w = ROS_window(self.master)
        self.w.top.configure(background="#6495ED")
        self.w.top.geometry("400x200")
        
        self.master.wait_window(self.w.top)

    
    # Count number of joints selected
    def select_joints(self):
        nose_enabled = self.check_nose.get()
        eyes_enabled = self.check_eyes.get()
        shoulders_enabled = self.check_shoulders.get()
        forefinger_enabled = self.check_forefinger.get()
        fingers_enabled = self.check_fingers.get()
        mouth_enabled=self.check_mouth.get()
        if nose_enabled:
            self.num_joints += 2
            self.joints[0, 0] = 1
        if eyes_enabled:
            self.num_joints += 4
            self.joints[1, 0] = 1
        if shoulders_enabled:
            self.num_joints += 4
            self.joints[2, 0] = 1
        if forefinger_enabled:
            self.num_joints += 2
            self.joints[3, 0] = 1
        if fingers_enabled:
            self.num_joints += 10
            self.joints[4, 0] = 1
        if mouth_enabled:
            self.num_joints += 4
            self.joints[5, 0] = 1
        if np.sum(self.joints, axis=0) != 0:
            self.btn_calib["state"] = "normal"
            self.btn_map["state"] = "normal"
            self.btn_custom["state"] = "normal"
            self.btn_start["state"] = "normal"
            print('Joints correctly selected.')

    def calibration(self):
        # start calibration dance - collect webcam data
        self.w = popupWindow(self.master, "You will now start calibration.")
        self.w.top.configure(background="#6495ED")
        self.w.top.geometry("300x100")
        
        self.master.wait_window(self.w.top)
        compute_calibration(self.calibPath, self.calib_duration, self.lbl_calib, self.num_joints, self.joints)
        self.btn_map["state"] = "normal"
        

    def train_map(self):
        # check whether calibration file exists first
        if os.path.isfile(self.calibPath + "Calib.txt"):
            self.w = popupWindow(self.master, "You will now train BoMI map")
            self.w.top.configure(background="#6495ED")
            self.w.top.geometry("300x100")
            self.master.wait_window(self.w.top)
            if self.check_pca.get():
                self.drPath = self.calibPath + 'PCA/'
                train_pca(self.calibPath, self.drPath)
                self.dr_mode = 'pca'
            elif self.check_ae.get():
                self.drPath = self.calibPath + 'AE/'
                train_ae(self.calibPath, self.drPath)
                self.dr_mode = 'ae'
            elif self.check_vae.get():
                self.drPath = self.calibPath + 'AE/'
                train_ae(self.calibPath, self.drPath)
                self.dr_mode = 'ae'
            elif self.check_tsne.get():
                self.drPath = self.calibPath + 'TSNE/'
                train_tsne(self.calibPath, self.drPath)
                self.dr_mode = 'tsne'
            self.btn_custom["state"] = "normal"
        else:
            self.w = popupWindow(self.master, "Perform calibration first.")
            self.master.wait_window(self.w.top)
            self.btn_map["state"] = "disabled"

    def customization(self):
        # check whether PCA/AE parameters have been saved
        if os.path.isfile(self.drPath + "weights1.txt"):
            # open customization window
            self.newWindow = tk.Toplevel(self.master)
            self.newWindow.geometry("1000x500")
            self.newWindow.title("Customization")
            self.app = CustomizationApplication(self.newWindow, self, drPath=self.drPath, num_joints=self.num_joints,
                                                joints=self.joints, dr_mode=self.dr_mode)
        else:
            self.w = popupWindow(self.master, "Compute BoMI map first.")
            self.master.wait_window(self.w.top)
            self.btn_custom["state"] = "disabled"

    def start(self):
        # check whether customization parameters have been saved
        if os.path.isfile(self.drPath + "offset_custom.txt"):
            # open pygame and start reaching task
            if not self.check_mouse.get():
                self.w = popupWindow(self.master, "You will now start practice.")
                self.w.top.configure(background="#6495ED")
                self.w.top.geometry("300x100")
                self.master.wait_window(self.w.top)
            else:
                key_thread = Thread(target=self.create_keyboard)
                key_thread.start() 
                
            s_thread=Thread(target=start_reaching,args=(self.drPath, self.lbl_tgt, self.num_joints, self.joints, self.dr_mode, self.check_mouse.get()))
            s_thread.start()        
            #start_reaching(self.drPath, self.lbl_tgt, self.num_joints, self.joints, self.dr_mode, self.check_mouse.get())
            
            # [ADD CODE HERE: one of the argument of start reaching should be [self.check_mouse]
            # to check in the checkbox is enable] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            self.w = popupWindow(self.master, "Perform customization first.")
            self.w.top.configure(background="#6495ED")
            self.w.top.geometry("300x100")
            self.master.wait_window(self.w.top)
            self.btn_start["state"] = "disabled"
    
    def create_keyboard(self):
        self.newWindow = tk.Toplevel(self.master)
        self.newWindow.geometry("1000x500")
        self.newWindow.title("Customization")
        self.app = VirtualKeyboardApplication(self.newWindow,self, drPath=self.drPath, num_joints=self.num_joints,
                                        joints=self.joints, dr_mode=self.dr_mode)


class CustomizationApplication(tk.Frame):
    """
    class that defines the customization tkinter window
    """

    def __init__(self, parent, mainTk, drPath, num_joints, joints, dr_mode):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.mainTk = mainTk
        self.drPath = drPath
        self.num_joints = num_joints
        self.joints = joints
        self.dr_mode = dr_mode
        self.font_size = 18
        self.parent.configure(background="#6495ED")
        self.lbl_rot = Label(parent, text='Rotation ',background="#6495ED")
        self.lbl_rot.config(font=("Arial", self.font_size))
        self.lbl_rot.grid(column=0, row=0, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_rot = Text(parent, width=10, height=1)
        self.txt_rot.config(font=("Arial", self.font_size))
        self.txt_rot.grid(column=1, row=0, pady=(40, 20))
        self.txt_rot.insert("1.0", '0')

        self.lbl_gx = Label(parent, text='Gain x ',background="#6495ED")
        self.lbl_gx.config(font=("Arial", self.font_size))
        self.lbl_gx.grid(column=0, row=1, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_gx = Text(parent, width=10, height=1)
        self.txt_gx.config(font=("Arial", self.font_size))
        self.txt_gx.grid(column=1, row=1, pady=(40, 20))
        self.txt_gx.insert("1.0", '1')

        self.lbl_gy = Label(parent, text='Gain y ',background="#6495ED")
        self.lbl_gy.config(font=("Arial", self.font_size))
        self.lbl_gy.grid(column=0, row=2, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_gy = Text(parent, width=10, height=1)
        self.txt_gy.config(font=("Arial", self.font_size))
        self.txt_gy.grid(column=1, row=2, pady=(40, 20))
        self.txt_gy.insert("1.0", '1')

        self.lbl_ox = Label(parent, text='Offset x ',background="#6495ED")
        self.lbl_ox.config(font=("Arial", self.font_size))
        self.lbl_ox.grid(column=0, row=3, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_ox = Text(parent, width=10, height=1)
        self.txt_ox.config(font=("Arial", self.font_size))
        self.txt_ox.grid(column=1, row=3, pady=(40, 20))
        self.txt_ox.insert("1.0", '0')

        self.lbl_oy = Label(parent, text='Offset y ',background="#6495ED")
        self.lbl_oy.config(font=("Arial", self.font_size))
        self.lbl_oy.grid(column=0, row=4, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_oy = Text(parent, width=10, height=1)
        self.txt_oy.config(font=("Arial", self.font_size))
        self.txt_oy.grid(column=1, row=4, pady=(40, 20))
        self.txt_oy.insert("1.0", '0')

        self.btn_save = Button(parent, text="Save parameters", command=self.save_parameters, borderwidth=3, relief="solid", bg='#006600',fg='black')
        self.btn_save.config(font=("Arial", self.font_size))
        self.btn_save.grid(column=2, row=1, sticky='nesw', padx=(80, 0), pady=(40, 20))

        self.btn_start = Button(parent, text="Start", command=self.customization, borderwidth=3, relief="solid", bg='#006600',fg='black')
        self.btn_start.config(font=("Arial", self.font_size))
        self.btn_start.grid(column=2, row=2, sticky='nesw', padx=(80, 0), pady=(40, 20))

        self.btn_close = Button(parent, text="Close", command=parent.destroy, bg='red')
        self.btn_close.config(font=("Arial", self.font_size))
        self.btn_close.grid(column=2, row=3, sticky='nesw', padx=(80, 0), pady=(40, 20))

    # functions to retrieve values of textbox programmatically
    def retrieve_txt_rot(self):
        return self.txt_rot.get("1.0", "end-1c")

    def retrieve_txt_gx(self):
        return self.txt_gx.get("1.0", "end-1c")

    def retrieve_txt_gy(self):
        return self.txt_gy.get("1.0", "end-1c")

    def retrieve_txt_ox(self):
        return self.txt_ox.get("1.0", "end-1c")

    def retrieve_txt_oy(self):
        return self.txt_oy.get("1.0", "end-1c")

    def customization(self):
        initialize_customization(self, self.dr_mode, self.drPath, self.num_joints, self.joints)

    def save_parameters(self):
        save_parameters(self, self.drPath)
        self.parent.destroy()
        self.mainTk.btn_start["state"] = "normal"


class popupWindow(object):
    """
    class that defines the popup tkinter window
    """

    def __init__(self, master, msg):
        self.lbl = Label(top, text=msg,background="#6495ED")
        self.lbl.pack()
        self.btn = Button(top, text='Ok', command=self.cleanup, borderwidth=3, relief="solid", bg='#006600',fg='black')
        self.btn.pack()

    def cleanup(self):
        self.top.destroy()
        

class ROS_window(object):
    """
    class that defines the popup ROS window
    """
    

    def __init__(self, master):
        self.targetx = 0.0
        self.targety = 0.0
        self.msgtwist = Twist()
        self.msgodometry = Odometry()
    	
        self.pub = rospy.Publisher("/cmd_vel",Twist,1000)
        self.sub = rospy.Subscriber("/odom", Odometry, self.odomclbk)
        

        top = self.top = tk.Toplevel(master)
        self.lbl = Label(top, text="Choose target",background="#6495ED")
        self.lbl.pack()
        self.btn1 = Button(top, text='1', command=lambda:self.order(1, 5), borderwidth=3, relief="solid", bg='#006600',fg='black')
        self.btn2 = Button(top, text='2', command=lambda:self.order(-2, 7), borderwidth=3, relief="solid", bg='#006600',fg='black')
        
        self.btn4 = Button(top, text='4', command=lambda:self.order(-2, -5), borderwidth=3, relief="solid", bg='#006600',fg='black')
        self.btn3 = Button(top, text='3', command=lambda:self.order(0, -3), borderwidth=3, relief="solid", bg='#006600',fg='black') 
        self.btn5 = Button(top, text='close', command=self.cleanup, borderwidth=3, relief="solid", bg='#006600',fg='black')

        self.btn1.pack()
        self.btn2.pack()
        self.btn3.pack()
        self.btn4.pack()
        self.btn5.pack()
        
    def order(self,target_numx, target_numy):
        self.targetx = target_numx
        self.targety = target_numy
        print ("Order")
    
    def cleanup(self):
        self.top.destroy()  
        
        #self.top.destroy()
        
    def odomclbk(self, msgodometry):
        k = 5
        actualx = msgodometry.pose.pose.position.x
        actualy = msgodometry.pose.pose.position.y
        print (actualx, actualy)
        distance = sqrt(pow(self.targetx-actualx,2)+pow(self.targety-actualy,2) )

        if (distance <= 0.1):
            rospy.loginfo("target reached")

        self.msgtwist.linear.x = k*(self.targetx-actualx)
        self.msgtwist.linear.y = k*(self.targety-actualy)
        self.pub.publish(self.msgtwist)



def compute_calibration(drPath, calib_duration, lbl_calib, num_joints, joints):
    """
    function called to collect calibration data from webcam
    :param drPath: path to save calibration file
    :param calib_duration: duration of calibration as read by the textbox in the main window
    :param lbl_calib: label in the main window that shows calibration time remaining
    :return:
    """
    # Create object of openCV and Reaching (needed for terminating mediapipe thread)
    cap = cv2.VideoCapture(0)
    r = Reaching()

    # The clock will be used to control how fast the screen updates. Stopwatch to count calibration time elapsed
    clock = pygame.time.Clock()
    timer_calib = StopWatch()

    # initialize MediaPipe Pose
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                    smooth_landmarks=False)

    # initialize lock for avoiding race conditions in threads
    lock = Lock()

    # global variable accessed by main and mediapipe threads that contains the current vector of body landmarks
    global body
    body = np.zeros((num_joints,))  # initialize global variable
    body_calib = []  # initialize local variable (list of body landmarks during calibration)

    # start thread for OpenCV. current frame will be appended in a queue in a separate thread
    q_frame = queue.Queue()
    cal = 1  # if cal==1 (meaning during calibration) the opencv thread will display the image
    opencv_thread = Thread(target=get_data_from_camera, args=(cap, q_frame, r, cal))
    opencv_thread.start()
    print("openCV thread started in calibration.")

    # initialize thread for mediapipe operations
    mediapipe_thread = Thread(target=mediapipe_forwardpass,
                              args=(holistic, mp_holistic, lock, q_frame, r, num_joints, joints))
    mediapipe_thread.start()
    print("mediapipe thread started in calibration.")

    # start the timer for calibration
    timer_calib.start()

    print("main thread: Starting calibration...")

    while not r.is_terminated:

        if timer_calib.elapsed_time > calib_duration:
            r.is_terminated = True

        # get current value of body
        body_calib.append(np.copy(body))

        # update time elapsed label
        time_remaining = int((calib_duration - timer_calib.elapsed_time) / 1000)
        lbl_calib.configure(text='Calibration time: ' + str(time_remaining))
        lbl_calib.update()

        # --- Limit to 50 frames per second
        clock.tick(50)

    # Stop the game engine and release the capture
    holistic.close()
    print("pose estimation object released in calibration.")
    cap.release()
    cv2.destroyAllWindows()
    print("openCV object released in calibration.")

    # print calibration file
    body_calib = np.array(body_calib)
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    np.savetxt(drPath + "Calib.txt", body_calib)

    print('Calibration finished. You can now train BoMI forward map.')


def train_pca(calibPath, drPath):
    """
    function to train BoMI forward map - PCA
    :param drPath: path to save BoMI forward map
    :return:
    """
    global model
    r = Reaching()
    # read calibration file and remove all the initial zero rows
    xp = list(pd.read_csv(calibPath + 'Calib.txt', sep=' ', header=None).values)
    x = [i for i in xp if all(i)]
    x = np.array(x)
    model='pca'
    # randomly shuffle input
    np.random.shuffle(x)

    # define train/test split
    thr = 80
    split = int(len(x) * thr / 100)
    train_x = x[0:split, :]
    test_x = x[split:, :]

    # initialize object of class PCA
    n_pc = 2
    PCA = PrincipalComponentAnalysis(n_pc)

    # train PCA
    pca, train_x_rec, train_pc, test_x_rec, test_pc = PCA.train_pca(train_x, x_test=test_x)
    print('PCA has been trained.')

    # save weights and biases
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    np.savetxt(drPath + "weights1.txt", pca.components_[:, :2])


    print('BoMI forward map (PCA parameters) has been saved.')

    # compute train/test VAF
    print(f'Training VAF: {compute_vaf(train_x, train_x_rec)}')
    print(f'Test VAF: {compute_vaf(test_x, test_x_rec)}')

    # normalize latent space to fit the monitor coordinates
    # Applying rotation
    train_pc = np.dot(train_x, pca.components_[:, :2])
    rot = 0
    train_pc[0] = train_pc[0] * np.cos(np.pi / 180 * rot) - train_pc[1] * np.sin(np.pi / 180 * rot)
    train_pc[1] = train_pc[0] * np.sin(np.pi / 180 * rot) + train_pc[1] * np.cos(np.pi / 180 * rot)
    # Applying scale
    scale = [r.width / np.ptp(train_pc[:, 0]), r.height / np.ptp(train_pc[:, 1])]
    train_pc = train_pc * scale
    # Applying offset
    off = [r.width / 2 - np.mean(train_pc[:, 0]), r.height / 2 - np.mean(train_pc[:, 1])]
    train_pc = train_pc + off

    # Plot latent space
    plt.figure()
    plt.scatter(train_pc[:, 0], train_pc[:, 1], c='green', s=20)
    plt.title('Projections in workspace')
    plt.axis("equal")

    # save AE scaling values
    with open(drPath + "rotation_dr.txt", 'w') as f:
        print(rot, file=f)
    np.savetxt(drPath + "scale_dr.txt", scale)
    np.savetxt(drPath + "offset_dr.txt", off)

    print('PCA scaling values has been saved. You can continue with customization.')

def train_tsne(calibPath, drPath):
    """
    function to train BoMI forward map - TSNE
    :param drPath: path to save BoMI forward map
    :return:
    """
    global model
    model='tsne'
    r = Reaching()
    # read calibration file and remove all the initial zero rows
    xp = list(pd.read_csv(calibPath + 'Calib.txt', sep=' ', header=None).values)
    x = [i for i in xp if all(i)]
    x = np.array(x)
    print(x)
    # train PCA
    tsne = TSNE(n_components=2, learning_rate='auto',  init='pca')
    print('TSNE has been trained.')
    #print(tsne.embedding_.T)
    X_embedded=tsne.fit_transform(x)
    # save weights and biases
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    np.savetxt(drPath + "weights1.txt", x)
    
   
    print('BoMI forward map (PCA parameters) has been saved.')


    # normalize latent space to fit the monitor coordinates
    # Applying rotation
    train_pc=X_embedded
    rot = 0
    train_pc[0] = train_pc[0] * np.cos(np.pi / 180 * rot) - train_pc[1] * np.sin(np.pi / 180 * rot)
    train_pc[1] = train_pc[0] * np.sin(np.pi / 180 * rot) + train_pc[1] * np.cos(np.pi / 180 * rot)
    # Applying scale
    scale = [r.width / np.ptp(train_pc[:, 0]), r.height / np.ptp(train_pc[:, 1])]
    train_pc = train_pc * scale
    # Applying offset
    off = [r.width / 2 - np.mean(train_pc[:, 0]), r.height / 2 - np.mean(train_pc[:, 1])]
    train_pc = train_pc + off

    # Plot latent space
    plt.figure()
    plt.scatter(train_pc[:, 0], train_pc[:, 1], c='green', s=20)
    plt.title('Projections in workspace')
    plt.axis("equal")

    # save AE scaling values
    with open(drPath + "rotation_dr.txt", 'w') as f:
        print(rot, file=f)
    np.savetxt(drPath + "scale_dr.txt", scale)
    np.savetxt(drPath + "offset_dr.txt", off)

    

def train_ae(calibPath, drPath):
    """
    function to train BoMI forward map
    :param drPath: path to save BoMI forward map
    :return:
    """
    global model
    model='ae'
    r = Reaching()

    # Autoencoder parameters
    n_steps = 3001
    lr = 0.02
    cu = 2
    nh1 = 6
    activ = "tanh"

    # read calibration file and remove all the initial zero rows
    xp = list(pd.read_csv(calibPath + 'Calib.txt', sep=' ', header=None).values)
    x = [i for i in xp if all(i)]
    x = np.array(x)

    # randomly shuffle input
    np.random.shuffle(x)

    # define train/test split
    thr = 80
    split = int(len(x) * thr / 100)
    train_x = x[0:split, :]
    test_x = x[split:, :]

    # initialize object of class Autoencoder
    AE = Autoencoder(n_steps, lr, cu, activation=activ, nh1=nh1, seed=0)

    # train AE network
    history, ws, bs, train_x_rec, train_cu, test_x_rec, test_cu = AE.train_network(train_x, x_test=test_x)
    # history, ws, bs, train_x_rec, train_cu, test_x_rec, test_cu = AE.train_vae(train_x, beta=0.00035, x_test=test_x)
    print('AE has been trained.')

    # save weights and biases
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    for layer in range(3):
        np.savetxt(drPath + "weights" + str(layer + 1) + ".txt", ws[layer])
        np.savetxt(drPath + "biases" + str(layer + 1) + ".txt", bs[layer])

    print('BoMI forward map (AE parameters) has been saved.')

    # compute train/test VAF
    print(f'Training VAF: {compute_vaf(train_x, train_x_rec)}')
    print(f'Test VAF: {compute_vaf(test_x, test_x_rec)}')

    # normalize latent space to fit the monitor coordinates
    # Applying rotation
    rot = 0
    train_cu[0] = train_cu[0] * np.cos(np.pi / 180 * rot) - train_cu[1] * np.sin(np.pi / 180 * rot)
    train_cu[1] = train_cu[0] * np.sin(np.pi / 180 * rot) + train_cu[1] * np.cos(np.pi / 180 * rot)
    # Applying scale
    scale = [r.width / np.ptp(train_cu[:, 0]), r.height / np.ptp(train_cu[:, 1])]
    train_cu = train_cu * scale
    # Applying offset
    off = [r.width / 2 - np.mean(train_cu[:, 0]), r.height / 2 - np.mean(train_cu[:, 1])]
    train_cu = train_cu + off

    # Plot latent space
    plt.figure()
    plt.scatter(train_cu[:, 0], train_cu[:, 1], c='green', s=20)
    plt.title('Projections in workspace')
    plt.axis("equal")

    # save AE scaling values
    with open(drPath + "rotation_dr.txt", 'w') as f:
        print(rot, file=f)
    np.savetxt(drPath + "scale_dr.txt", scale)
    np.savetxt(drPath + "offset_dr.txt", off)

    print('AE scaling values has been saved. You can continue with customization.')

    


def load_bomi_map(dr_mode, drPath):
    if dr_mode == 'pca':
        map = pd.read_csv(drPath + 'weights1.txt', sep=' ', header=None).values
    elif dr_mode == 'tsne':

        map = pd.read_csv(drPath + 'weights1.txt', sep=' ', header=None).values
        print('MAP')
        print(map)
    elif dr_mode == 'ae':
        ws = []
        bs = []
        ws.append(pd.read_csv(drPath + 'weights1.txt', sep=' ', header=None).values)
        ws.append(pd.read_csv(drPath + 'weights2.txt', sep=' ', header=None).values)
        ws.append(pd.read_csv(drPath + 'weights3.txt', sep=' ', header=None).values)
        bs.append(pd.read_csv(drPath + 'biases1.txt', sep=' ', header=None).values)
        bs[0] = bs[0].reshape((bs[0].size,))
        bs.append(pd.read_csv(drPath + 'biases2.txt', sep=' ', header=None).values)
        bs[1] = bs[1].reshape((bs[1].size,))
        bs.append(pd.read_csv(drPath + 'biases3.txt', sep=' ', header=None).values)
        bs[2] = bs[2].reshape((bs[2].size,))

        map = (ws, bs)

    return map


def initialize_customization(self, dr_mode, drPath, num_joints, joints):
    """
    initialize objects needed for online cursor control. Start all the customization threads as well
    :param self: CustomizationApplication tkinter Frame. needed to retrieve textbox values programmatically
    :param drPath: path to load the BoMI forward map
    :return:
    """

    # Create object of openCV, Reaching class and filter_butter3
    global prev_cx, prev_cy
    cap = cv2.VideoCapture(0)
    r = Reaching()
    filter_curs = FilterButter3("lowpass_4")

    # initialize target position
    reaching_functions.initialize_targets(r)

    # load BoMI forward map parameters for converting body landmarks into cursor coordinates
    map = load_bomi_map(dr_mode, drPath)

    # initialize MediaPipe Pose
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                    smooth_landmarks=False)

    # load scaling values saved after training AE for covering entire monitor workspace
    rot = pd.read_csv(drPath + 'rotation_dr.txt', sep=' ', header=None).values
    scale = pd.read_csv(drPath + 'scale_dr.txt', sep=' ', header=None).values
    scale = np.reshape(scale, (scale.shape[0],))
    off = pd.read_csv(drPath + 'offset_dr.txt', sep=' ', header=None).values
    off = np.reshape(off, (off.shape[0],))

    # initialize lock for avoiding race conditions in threads
    lock = Lock()

    # global variable accessed by main and mediapipe threads that contains the current vector of body landmarks
    global body
    body = np.zeros((num_joints,))  # initialize global variable

    # start thread for OpenCV. current frame will be appended in a queue in a separate thread
    q_frame = queue.Queue()
    cal = 0
    opencv_thread = Thread(target=get_data_from_camera, args=(cap, q_frame, r, cal))
    opencv_thread.start()
    print("openCV thread started in customization.")

    # initialize thread for mediapipe operations
    mediapipe_thread = Thread(target=mediapipe_forwardpass,
                              args=(holistic, mp_holistic, lock, q_frame, r, num_joints, joints))
    mediapipe_thread.start()
    print("mediapipe thread started in customization.")

    # Define some colors
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    CURSOR = (0.19 * 255, 0.65 * 255, 0.4 * 255)

    pygame.init()

    # The clock will be used to control how fast the screen updates
    clock = pygame.time.Clock()

    # Open a new window
    size = (r.width, r.height)
    screen = pygame.display.set_mode(size)
    # screen = pygame.display.toggle_fullscreen()

    # -------- Main Program Loop -----------
    while not r.is_terminated:
        # --- Main event loop
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                r.is_terminated = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                    r.is_terminated = True
                if event.key == pygame.K_SPACE:  # Pressing the space Key will click the mouse
                    pyautogui.click(r.crs_x, r.crs_y)

        if not r.is_paused:
            # Copy old cursor position
            r.old_crs_x = r.crs_x
            r.old_crs_y = r.crs_y

            # get current value of body
            r.body = np.copy(body)

            # apply BoMI forward map to body vector to obtain cursor position
            r.crs_x, r.crs_y = reaching_functions.update_cursor_position_custom(r.body, map, rot, scale, off, model)

            # Apply extra customization according to textbox values (try/except allows to catch invalid inputs)
            try:
                rot_custom = int(self.retrieve_txt_rot())
            except:
                rot_custom = 0
            try:
                gx_custom = float(self.retrieve_txt_gx())
            except:
                gx_custom = 1.0
            try:
                gy_custom = float(self.retrieve_txt_gy())
            except:
                gy_custom = 1.0
            try:
                ox_custom = int(self.retrieve_txt_ox())
            except:
                ox_custom = 0
            try:
                oy_custom = int(self.retrieve_txt_oy())
            except:
                oy_custom = 0

            # Applying rotation
            r.crs_x = r.crs_x * np.cos(np.pi / 180 * rot_custom) - r.crs_y * np.sin(np.pi / 180 * rot_custom)
            r.crs_y = r.crs_x * np.sin(np.pi / 180 * rot_custom) + r.crs_y * np.cos(np.pi / 180 * rot_custom)
            # Applying scale
            r.crs_x = r.crs_x * gx_custom
            r.crs_y = r.crs_y * gy_custom
            # Applying offset
            r.crs_x = r.crs_x + ox_custom
            r.crs_y = r.crs_y + oy_custom

            # Limit cursor workspace
            if r.crs_x >= r.width:
                r.crs_x = r.width
            if r.crs_x <= 0:
                r.crs_x = 0
            if r.crs_y >= r.height:
                r.crs_y = 0
            if r.crs_y <= 0:
                r.crs_y = r.height

            # Filter the cursor
            r.crs_x, r.crs_y = reaching_functions.filter_cursor(r, filter_curs)

            # Set target position to update the GUI
            reaching_functions.set_target_reaching_customization(r)

            # First, clear the screen to black. In between screen.fill and pygame.display.flip() all the draw
            screen.fill(BLACK)

            # draw cursor
            pygame.draw.circle(screen, CURSOR, (int(r.crs_x), int(r.crs_y)), r.crs_radius)

            # draw each test target
            for i in range(8):
                tgt_x = r.tgt_x_list[r.list_tgt[i]]
                tgt_y = r.tgt_y_list[r.list_tgt[i]]
                pygame.draw.circle(screen, GREEN, (int(tgt_x), int(tgt_y)), r.tgt_radius, 2)

            # --- update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 50 frames per second
            clock.tick(50)

    # Once we have exited the main program loop, stop the game engine and release the capture
    pygame.quit()
    print("game engine object released in customization.")
    holistic.close()
    print("pose estimation object released terminated in customization.")
    cap.release()
    cv2.destroyAllWindows()
    print("openCV object released in customization.")

def save_parameters(self, drPath):
    """
    function to save customization values
    :param self: CustomizationApplication tkinter Frame. needed to retrieve textbox values programmatically
    :param drPath: path where to load the BoMI forward map
    :return:
    """
    # retrieve values stored in the textbox
    rot = int(self.retrieve_txt_rot())
    gx_custom = float(self.retrieve_txt_gx())
    gy_custom = float(self.retrieve_txt_gy())
    scale = [gx_custom, gy_custom]
    ox_custom = int(self.retrieve_txt_ox())
    oy_custom = int(self.retrieve_txt_oy())
    off = [ox_custom, oy_custom]

    # save customization values
    with open(drPath + "rotation_custom.txt", 'w') as f:
        print(rot, file=f)
    np.savetxt(drPath + "scale_custom.txt", scale)
    np.savetxt(drPath + "offset_custom.txt", off)

    print('Customization values have been saved. You can continue with practice.')

# [ADD CODE HERE: check_mouse as function input]



class VirtualKeyboardApplication(tk.Frame):
    """
    class that defines 
    """
    
    def __init__(self, parent, mainTk, drPath, num_joints, joints, dr_mode):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.mainTk = mainTk
        self.drPath = drPath
        self.num_joints = num_joints
        self.joints = joints
        self.dr_mode = dr_mode
        self.font_size = 18
        self.equation = tk.StringVar()
        self.exp = " "
        self.parent.configure(background="#6495ED")
        
        self.Dis_entry = ttk.Entry(parent,state= 'readonly',textvariable = self.equation)
        self.Dis_entry.grid(rowspan= 1 , columnspan = 100, ipadx = 999 , ipady = 20)
     
        self.q = ttk.Button(parent,text = 'Q' , width = 6, command = lambda :  self.press('Q'))
        self.q.grid(row = 1 , column = 0, ipadx = 6 , ipady = 10)

        self.w = ttk.Button(parent,text = 'W' , width = 6, command =lambda :  self.press('W'))
        self.w.grid(row = 1 , column = 1, ipadx = 6 , ipady = 10)

        self.E = ttk.Button(parent,text = 'E' , width = 6, command = lambda : self.press('E'))
        self.E.grid(row = 1 , column = 2, ipadx = 6 , ipady = 10)

        self.R = ttk.Button(parent,text = 'R' , width = 6, command = lambda : self.press('R'))
        self.R.grid(row = 1 , column = 3, ipadx = 6 , ipady = 10)

        self.T = ttk.Button(parent,text = 'T' , width = 6, command = lambda : self.press('T'))
        self.T.grid(row = 1 , column = 4, ipadx = 6 , ipady = 10)

        self.Y = ttk.Button(parent,text = 'Y' , width = 6, command = lambda : self.press('Y'))
        self.Y.grid(row = 1 , column = 5, ipadx = 6 , ipady = 10)

        self.U = ttk.Button(parent,text = 'U' , width = 6, command = lambda : self.press('U'))
        self.U.grid(row = 1 , column = 6, ipadx = 6 , ipady = 10)

        self.I = ttk.Button(parent,text = 'I' , width = 6, command = lambda : self.press('I'))
        self.I.grid(row = 1 , column = 7, ipadx = 6 , ipady = 10)

        self.O = ttk.Button(parent,text = 'O' , width = 6, command = lambda : self.press('O'))
        self.O.grid(row = 1 , column = 8, ipadx = 6 , ipady = 10)

        self.P = ttk.Button(parent,text = 'P' , width = 6, command =lambda :  self.press('P'))
        self.P.grid(row = 1 , column = 9, ipadx = 6 , ipady = 10)

        self.cur = ttk.Button(parent,text = '{' , width = 6, command = lambda : self.press('{'))
        self.cur.grid(row = 1 , column = 10 , ipadx = 6 , ipady = 10)

        self.cur_c = ttk.Button(parent,text = '}' , width = 6, command =lambda :  self.press('}'))
        self.cur_c.grid(row = 1 , column = 11, ipadx = 6 , ipady = 10)

        self.back_slash = ttk.Button(parent,text = '\\' , width = 6, command = lambda : self.press('\\'))
        self.back_slash.grid(row = 1 , column = 12, ipadx = 6 , ipady = 10)


        self.clearbtn = ttk.Button(parent,text = 'Clear' , width = 6, command =lambda :  self.clear(' '))
        self.clearbtn.grid(row = 1 , column = 13, ipadx = 20 , ipady = 10)

        # Second Line Button



        self.A = ttk.Button(parent,text = 'A' , width = 6, command = lambda : self.press('A'))
        self.A.grid(row = 2 , column = 0, ipadx = 6 , ipady = 10)



        self.S = ttk.Button(parent,text = 'S' , width = 6, command = lambda : self.press('S'))
        self.S.grid(row = 2 , column = 1, ipadx = 6 , ipady = 10)

        self.D = ttk.Button(parent,text = 'D' , width = 6, command = lambda : self.press('D'))
        self.D.grid(row = 2 , column = 2, ipadx = 6 , ipady = 10)

        self.F = ttk.Button(parent,text = 'F' , width = 6, command = lambda : self.press('F'))
        self.F.grid(row = 2 , column = 3, ipadx = 6 , ipady = 10)


        self.G = ttk.Button(parent,text = 'G' , width = 6, command = lambda : self.press('G'))
        self.G.grid(row = 2 , column = 4, ipadx = 6 , ipady = 10)


        self.H = ttk.Button(parent,text = 'H' , width = 6, command =lambda :  self.press('H'))
        self.H.grid(row = 2 , column = 5, ipadx = 6 , ipady = 10)


        self.J = ttk.Button(parent,text = 'J' , width = 6, command =lambda :  self.press('J'))
        self.J.grid(row = 2 , column = 6, ipadx = 6 , ipady = 10)


        self.K = ttk.Button(parent,text = 'K' , width = 6, command =lambda :  self.press('K'))
        self.K.grid(row = 2 , column = 7, ipadx = 6 , ipady = 10)

        self.L = ttk.Button(parent,text = 'L' , width = 6, command = lambda : self.press('L'))
        self.L.grid(row = 2 , column = 8, ipadx = 6 , ipady = 10)


        self.semi_co = ttk.Button(parent,text = ';' , width = 6, command = lambda : self.press(';'))
        self.semi_co.grid(row = 2 , column = 9, ipadx = 6 , ipady = 10)


        self.d_colon = ttk.Button(parent,text = '"' , width = 6, command = lambda : self.press('"'))
        self.d_colon.grid(row = 2 , column = 10, ipadx = 6 , ipady = 10)


        self.enter = ttk.Button(parent,text = 'Enter' , width = 6, command = lambda : self.action(' Next Line : '))
        self.enter.grid(row = 2 , columnspan = 75, ipadx = 85 , ipady = 10)

        # third line Button

        self.Z = ttk.Button(parent,text = 'Z' , width = 6, command = lambda : self.press('Z'))
        self.Z.grid(row = 3 , column = 0, ipadx = 6 , ipady = 10)


        self.X = ttk.Button(parent,text = 'X' , width = 6, command = lambda :  self.press('X'))
        self.X.grid(row = 3 , column = 1, ipadx = 6 , ipady = 10)


        self.C = ttk.Button(parent,text = 'C' , width = 6, command =lambda :  self.press('C'))
        self.C.grid(row = 3 , column = 2, ipadx = 6 , ipady = 10)


        self.V = ttk.Button(parent,text = 'V' , width = 6, command = lambda : self.press('V'))
        self.V.grid(row = 3 , column = 3, ipadx = 6 , ipady = 10)

        self.B = ttk.Button(parent, text= 'B' , width = 6 , command =lambda :  self.press('B'))
        self.B.grid(row = 3 , column = 4 , ipadx = 6 ,ipady = 10)


        self.N = ttk.Button(parent,text = 'N' , width = 6, command = lambda : self.press('N'))
        self.N.grid(row = 3 , column = 5, ipadx = 6 , ipady = 10)


        self.M = ttk.Button(parent,text = 'M' , width = 6, command = lambda : self.press('M'))
        self.M.grid(row = 3 , column = 6, ipadx = 6 , ipady = 10)


        self.left = ttk.Button(parent,text = '<' , width = 6, command = lambda : self.press('<'))
        self.left.grid(row = 3 , column = 7, ipadx = 6 , ipady = 10)


        self.right = ttk.Button(parent,text = '>' , width = 6, command =lambda :  self.press('>'))
        self.right.grid(row = 3 , column = 8, ipadx = 6 , ipady = 10)


        self.slas = ttk.Button(parent,text = '/' , width = 6, command =lambda :  self.press('/'))
        self.slas.grid(row = 3 , column = 9, ipadx = 6 , ipady = 10)


        self.q_mark = ttk.Button(parent,text = '?' , width = 6, command =lambda :  self.press('?'))
        self.q_mark.grid(row = 3 , column = 10, ipadx = 6 , ipady = 10)


        self.coma = ttk.Button(parent,text = ',' , width = 6, command =lambda :  self.press(','))
        self.coma.grid(row = 3 , column = 11, ipadx = 6 , ipady = 10)

        self.dot = ttk.Button(parent,text = '.' , width = 6, command =lambda :  self.press('.'))
        self.dot.grid(row = 3 , column = 12, ipadx = 6 , ipady = 10)

        self.shift = ttk.Button(parent,text = 'Shift' , width = 6, command = lambda : self.press('Shift'))
        self.shift.grid(row = 3 , column = 13, ipadx = 20 , ipady = 10)

        #Fourth Line Button


        self.ctrl = ttk.Button(parent,text = 'Ctrl' , width = 6, command = lambda : self.press('Ctrl'))
        self.ctrl.grid(row = 4 , column = 0, ipadx = 6 , ipady = 10)


        self.Fn = ttk.Button(parent,text = 'Fn' , width = 6, command =lambda :  self.press('Fn'))
        self.Fn.grid(row = 4 , column = 1, ipadx = 6 , ipady = 10)


        self.window = ttk.Button(parent,text = 'Window' , width = 6, command = lambda : self.press('Window'))
        self.window.grid(row = 4 , column = 2 , ipadx = 6 , ipady = 10)

        self.Alt = ttk.Button(parent,text = 'Alt' , width = 6, command = lambda : self.press('Alt'))
        self.Alt.grid(row = 4 , column = 3 , ipadx = 6 , ipady = 10)

        self.space = ttk.Button(parent,text = 'Space' , width = 6, command = lambda : self.press(' '))
        self.space.grid(row = 4 , columnspan = 14 , ipadx = 160 , ipady = 10)

        self.Alt_gr = ttk.Button(parent,text = 'Alt Gr' , width = 6, command = lambda : self.press('Alt Gr'))
        self.Alt_gr.grid(row = 4 , column = 10 , ipadx = 6 , ipady = 10)

        self.open_b = ttk.Button(parent,text = '(' , width = 6, command =lambda :  self.press('('))
        self.open_b.grid(row = 4 , column = 11 , ipadx = 6 , ipady = 10)

        self.close_b = ttk.Button(parent,text = ')' , width = 6, command = lambda : self.press(')'))
        self.close_b.grid(row = 4 , column = 12 , ipadx = 6 , ipady = 10)


        self.tap = ttk.Button(parent,text = 'Tab' , width = 6, command =lambda :  self.Tab(' Tab: '))
        self.tap.grid(row = 4 , column = 13 , ipadx = 20 , ipady = 10)
    
    # functions to retrieve values of textbox programmatically

    def press(self,num):
       
        self.exp=self.exp + str(num)
        self.equation.set(self.exp)
        print(self.exp)
        print(self.equation)
    # end 

    # function clear button

    def clear(self,num):
        self.exp = str(num)
        self.equation.set(self.exp)
        print(self.exp)
        print(self.equation)

    # end 

    # Enter Button Work Next line Function

    def action(self,num): 
        self.exp = str(num)
        self.equation.set(self.exp)
        print(self.exp)
        print(self.equation)

    # end function coding

    # Tab Button Function 

    def Tab(self,num): 
        self.exp = str(num)
        self.equation.set(self.exp)
        print(self.exp)
        print(self.equation)


def start_reaching(drPath, lbl_tgt, num_joints, joints, dr_mode, mouse_enabled):
    """
    function to perform online cursor control - practice
    :param drPath: path where to load the BoMI forward map and customization values
    :param check_mouse: tkinter Boolean value that triggers mouse control instead of reaching task
    :param lbl_tgt: label in the main window that shows number of targets remaining
    :return:
    """
    pygame.init()

    # [ADD CODE HERE] get value from checkbox - is mouse enabled? !!!!!!!!!!!!!!!!!!!
    global MASTER
    m_enabled = mouse_enabled
    flag = True
    ############################################################
    th=50
    # Define some colors
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    CURSOR = (0.19 * 255, 0.65 * 255, 0.4 * 255)

    # Create object of openCV, Reaching class and filter_butter3
    cap = cv2.VideoCapture(0)
    r = Reaching()
    filter_curs = FilterButter3("lowpass_4")

    # Open a new window
    timer_click=StopWatch()
    if not m_enabled:
        size = (r.width, r.height)
        screen = pygame.display.set_mode(size)
    else:
        timer_click.start()
    #screen = pygame.display.toggle_fullscreen()

    # The clock will be used to control how fast the screen updates
    clock = pygame.time.Clock()

    # Initialize stopwatch for counting time elapsed in the different states of the reaching
    timer_enter_tgt = StopWatch()
    timer_start_trial = StopWatch()
    timer_practice = StopWatch()
    
    # initialize targets and the reaching log file header
    reaching_functions.initialize_targets(r)
    reaching_functions.write_header(r)

    # load BoMI forward map parameters for converting body landmarks into cursor coordinates
    map = load_bomi_map(dr_mode, drPath)

    # initialize MediaPipe Pose
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                    smooth_landmarks=False)

    # load scaling values for covering entire monitor workspace
    rot_dr = pd.read_csv(drPath + 'rotation_dr.txt', sep=' ', header=None).values
    scale_dr = pd.read_csv(drPath + 'scale_dr.txt', sep=' ', header=None).values
    scale_dr = np.reshape(scale_dr, (scale_dr.shape[0],))
    off_dr = pd.read_csv(drPath + 'offset_dr.txt', sep=' ', header=None).values
    off_dr = np.reshape(off_dr, (off_dr.shape[0],))
    rot_custom = pd.read_csv(drPath + 'rotation_custom.txt', sep=' ', header=None).values
    scale_custom = pd.read_csv(drPath + 'scale_custom.txt', sep=' ', header=None).values
    scale_custom = np.reshape(scale_custom, (scale_custom.shape[0],))
    off_custom = pd.read_csv(drPath + 'offset_custom.txt', sep=' ', header=None).values
    off_custom = np.reshape(off_custom, (off_custom.shape[0],))

    # initialize lock for avoiding race conditions in threads
    lock = Lock()

    # global variable accessed by main and mediapipe threads that contains the current vector of body landmarks
    global body
    body = np.zeros((num_joints,))  # initialize global variable

    # start thread for OpenCV. current frame will be appended in a queue in a separate thread
    q_frame = queue.Queue()
    cal = 0
    opencv_thread = Thread(target=get_data_from_camera, args=(cap, q_frame, r, cal))
    opencv_thread.start()
    print("openCV thread started in practice.")

    # initialize thread for mediapipe operations
    mediapipe_thread = Thread(target=mediapipe_forwardpass,
                              args=(holistic, mp_holistic, lock, q_frame, r, num_joints, joints))
    mediapipe_thread.start()
    print("mediapipe thread started in practice.")

    # initialize thread for writing reaching log file
    wfile_thread = Thread(target=write_practice_files, args=(r, timer_practice))
    timer_practice.start()  # start the timer for PracticeLog
    wfile_thread.start()
    print("writing reaching log file thread started in practice.")

    print("cursor control thread is about to start...")

    # -------- Main Program Loop -----------
    while not r.is_terminated:
        # --- Main event loop
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                r.is_terminated = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                    r.is_terminated = True
                if event.key == pygame.K_p:  # Pressing the p Key will pause/resume the game
                    reaching_functions.pause_acquisition(r, timer_practice)
                if event.key == pygame.K_SPACE:  # Pressing the space Key will click the mouse
                    pyautogui.click(r.crs_x, r.crs_y)

        if not r.is_paused:
            # Copy old cursor position
            r.old_crs_x = r.crs_x
            r.old_crs_y = r.crs_y

            # get current value of body
            r.body = np.copy(body)

            # apply BoMI forward map to body vector to obtain cursor position.
            r.crs_x, r.crs_y = reaching_functions.update_cursor_position \
                (r.body, map, rot_dr, scale_dr, off_dr, rot_custom, scale_custom, off_custom, model)

            # Check if the crs is bouncing against any of the 4 walls:
            if r.crs_x >= r.width:
                r.crs_x = r.width
            if r.crs_x <= 0:
                r.crs_x = 0
            if r.crs_y >= r.height:
                r.crs_y = 0
            if r.crs_y <= 0:
                r.crs_y = r.height

            # Filter the cursor
            r.crs_x, r.crs_y = reaching_functions.filter_cursor(r, filter_curs)
            
            # if mouse checkbox was enabled do not draw the reaching GUI,
            # only change coordinates of the computer cursor !!!!!!!!!!!!!!!!!!!!!
            # [ADD CODE HERE] !!!!!!!!!!!!!!!!!!!!!
            if m_enabled:
                pyautogui.moveTo(r.crs_x, r.crs_y)
                print(timer_click.elapsed_time)
                print(((r.old_crs_x - r.crs_x)**2+(r.old_crs_y - r.crs_y)**2))
                if ((r.old_crs_x - r.crs_x)**2+(r.old_crs_y - r.crs_y)**2)>th:
                      timer_click.start()
                if timer_click.elapsed_time > 2000:
                      pyautogui.click(r.crs_x,r.crs_y)

            else:

                # Set target position to update the GUI
                reaching_functions.set_target_reaching(r)
                # First, clear the screen to black. In between screen.fill and pygame.display.flip() all the draw
                screen.fill(BLACK)
                # Do not show the cursor in the blind trials when the cursor is outside the home target
                if not r.is_blind:
                    # draw cursor
                    pygame.draw.circle(screen, CURSOR, (int(r.crs_x), int(r.crs_y)), r.crs_radius)

                # draw target. green if blind, state 0 or 1. yellow if notBlind and state 2
                if r.state == 0:  # green
                    pygame.draw.circle(screen, GREEN, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)
                elif r.state == 1:
                    pygame.draw.circle(screen, GREEN, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)
                elif r.state == 2:  # yellow
                    if r.is_blind:  # green again if blind trial
                        pygame.draw.circle(screen, GREEN, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)
                    else:  # yellow if not blind
                        pygame.draw.circle(screen, YELLOW, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)

                # Display scores:
                font = pygame.font.Font(None, 80)
                text = font.render(str(r.score), True, RED)
                screen.blit(text, (1250, 10))

                # --- update the screen with what we've drawn.
                pygame.display.flip()

                # After showing the cursor, check whether cursor is in the target
                reaching_functions.check_target_reaching(r, timer_enter_tgt)
                # Then check if cursor stayed in the target for enough time
                reaching_functions.check_time_reaching(r, timer_enter_tgt, timer_start_trial, timer_practice)

                # update label with number of targets remaining
                tgt_remaining = 248 - r.trial + 1
                lbl_tgt.configure(text='Remaining targets: ' + str(tgt_remaining))
                lbl_tgt.update()

                # --- Limit to 50 frames per second
                clock.tick(50)

    # Once we have exited the main program loop, stop the game engine and release the capture
    pygame.quit()
    print("game engine object released in practice.")
    # pose.close()
    holistic.close()
    print("pose estimation object released in practice.")
    cap.release()
    cv2.destroyAllWindows()
    print("openCV object released in practice.")


def get_data_from_camera(cap, q_frame, r, cal):
    '''
    function that runs in the thread to capture current frame and put it into the queue
    :param cap: object of OpenCV class
    :param q_frame: queue to store current frame
    :param r: object of Reaching class
    :return:
    '''
    while not r.is_terminated:
        if not r.is_paused:
            ret, frame = cap.read()
            q_frame.put(frame)
            # if cal == 1:
            #    cv2.imshow('current frame', frame)
    print('OpenCV thread terminated.')


def mediapipe_forwardpass(holistic, mp_holistic, lock, q_frame, r, num_joints, joints):
    """
    function that runs in the thread for estimating pose online
    :param pose: object of Mediapipe class used to predict poses
    :param mp_pose: object of Mediapipe class for extracting body landmarks
    :param lock: lock for avoiding race condition on body vector
    :param q_frame: queue where to append current webcam frame
    :param r: object of Reaching class
    :return:
    """
    global body
    while not r.is_terminated:
        if not r.is_paused:
            # not sure if we want to put try/catch here, just in case "ask forgiveness, not permission"
            # try:
            # get current frame from thread
            curr_frame = q_frame.get()
            body_list = []

            # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(curr_frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = holistic.process(image)
            
            if not results.pose_landmarks:
                continue
            if joints[0, 0] == 1:
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)
            if joints[1, 0] == 1:
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y)
            if joints[2, 0] == 1:
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
            if joints[3, 0] == 1 or joints[4, 0] == 1:
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
            if joints[4, 0] == 1:
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x)
                body_list.append(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y)
            if joints[5, 0] == 1:
                print('calibrated with mouth')
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x)
                body_list.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y)
            body_mp = np.array(body_list)
            q_frame.queue.clear()
            with lock:
                body = np.copy(body_mp)
            # except:
            #     print('Expection in mediapipe_forwardpass. Closing thread')
            #     r.is_terminated = True
    print('Mediapipe_forwardpass thread terminated.')


def write_practice_files(r, timer_practice):
    """
    function that runs in the thread for writing reaching log in a file
    :param r: object of Reaching class
    :param timer_practice: stopwatch that keeps track of elapsed time during reaching
    :return:
    """
    while not r.is_terminated:
        if not r.is_paused:
            starttime = time.time()

            log = str(timer_practice.elapsed_time) + "\t" + '\t'.join(map(str, r.body)) + "\t" + str(r.crs_x) + "\t" + \
                  str(r.crs_y) + "\t" + str(r.block) + "\t" + \
                  str(r.repetition) + "\t" + str(r.target) + "\t" + str(r.trial) + "\t" + str(r.state) + "\t" + \
                  str(r.comeback) + "\t" + str(r.is_blind) + "\t" + str(r.at_home) + "\t" + str(r.count_mouse) + "\t" + \
                  str(r.score) + "\n"

            with open(r.path_log + "PracticeLog.txt", "a") as file_log:
                file_log.write(log)

            # write @ 50 Hz
            time.sleep(0.033 - ((time.time() - starttime) % 0.033))

    print('Writing reaching log file thread terminated.')


# CODE STARTS HERE
if __name__ == "__main__":
    rospy.init_node('main_reaching')
    
    
    # initialize mainApplication tkinter window
    win = tk.Tk()
    win.configure(background="#6495ED")
    win.title("BoMI Settings")
    
  
    window_width = 1500
    window_height = 700

    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()

    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))

    win.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
   #MASTER = win
    MainApplication(win)

    # initiate Tkinter mainloop
    win.mainloop()
