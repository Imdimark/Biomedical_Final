# Biomedical robotics _ Last assignment
## Nardelli Alice _ Di Marco Giovanni _ Pietrasanta Iacopo _ Civetta Federico


In order to make the project compliant to ROS the original project is splitted in two folder and a third one is added, it is used to simulate an holonomic environment. 

### Requirements
  * Al these requirements must be satisfied: https://github.com/MoroMatteo/markerlessBoMI_FaMa/blob/main/README.md 
  
  In this case only the linux option must be used in order to be compliant to ROS and the step 7 is different in our case: the repository must be this one.
  We don't recommend to use a docker container, it should create issues with the camera input. 
  
  * ROS workspace with inside the three packages that are inside this repository
 
### How to launch

1. Download the repository: ``git clone https://github.com/Imdimark/Biomedical_Final``
2. Compile the packages: inside the workspace launch ``catkin_make``
3. start the ROSCORE: ``roscore &`` 
5. launch the code: ``rosrun biome main_reaching.py``

The ROS component will be launched when you choose the "ROS GAME" option. 


### ROS game Behaviour

![rosgraph](https://user-images.githubusercontent.com/78663960/159170794-b8ee8900-97d4-4e12-bd8d-e025b788a289.png)

Two windows are opened:
* the simulation environment by stage_ros
* a button panel with 4 possible options

Every button is associated with a specific position, so in total we have 4 possible position (x, y). 
Every time the goal is reached, the success status is printed in the screen. 

 




