
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:07:20 2019

@author: 14791
"""

import numpy as np

from bipeRotor_vrep import SpaceRobot3link
import matplotlib.pyplot as plt

# Environment

env = SpaceRobot3link()
#torch.manual_seed(args.seed)
#np.random.seed(args.seed)
#env.seed(args.seed)
# Agent

# Memory

total_numsteps = 0
updates = 0
totalr=[]
step=60


target_z=1
target_x=1
target_y=1


kp_xx=0.11
kd_xx=0.11
kp_yy=-0.5
kd_yy=-0.5

target_joint_r=0
target_joint_l=0
target_joint_v1=0

target_yaw=0
target_roll=0
target_pitch=0


Kp_y=2
Kd_y=1
Kp_r=180
Kd_r=50
Kp_p=5
Kd_p=5
Kp_z=100
kd_z=30
Kp_joint=-0.025
Kd_joint=-0.005
for i_episode in range(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    r=[]
    state = env.reset()
    control_information=np.zeros([step,4])
    state_inf=np.zeros([step,21])
    
    for i_epoch in range(step):
        #完全随机动作序列
        z=state[11]

        v_z=state[14]
        p_z=Kp_z*(target_z-z)-kd_z*v_z+50
        
        target_roll=kp_yy*(target_y-state[10])-kd_yy*state[13]
        target_pitch=kp_xx*(target_x-state[9])-kd_xx*state[12]
        
        output_yaw=Kp_y*(target_yaw-state[8])+Kd_y*(-state[17])
        putput_roll=Kp_r*(target_roll-state[6])+Kd_r*(-state[15])
        output_pitch=Kp_p*(target_pitch-state[7])+Kd_p*(-state[16])
        
        for i in range(5):
            joint1=state[0]
            joint2=state[1]
            joint_v1=state[2]
            joint_v2=state[3]
            target_joint_r=output_yaw-output_pitch
            target_joint_l=-output_yaw-output_pitch
            

            
            target_joint_r=np.clip(-0.4,0.4,target_joint_r)
            target_joint_l=np.clip(-0.4,0.4,target_joint_l)
            
            p_joint_r=Kp_joint*(target_joint_r-joint1)+Kd_joint*(-joint_v1)
            p_joint_l=Kp_joint*(target_joint_l-joint2)+Kd_joint*(-joint_v2) 
            
            
            
            output_t_r=p_z-putput_roll
            output_t_l=p_z+putput_roll
            
            
            
            action =np.array([p_joint_r,p_joint_l,output_t_r,output_t_l])
    
            #连续正弦动作序列
            
            next_state, reward, done, _ = env.step(action) # Step
            control_information[i_epoch]=action
            state_inf[i_epoch]=next_state
            state=next_state
        
env.close()

plt.plot(0.01*np.arange(len(state_inf[:,0])),state_inf[:,0],label='joint1')
plt.plot(0.01*np.arange(len(state_inf[:,1])),state_inf[:,1],label='joint2')
plt.ylabel('angle/rad')
plt.xlabel('time/s')
#plt.plot(np.arange(len(state_inf[:,2])),state_inf[:,11],label='z')
plt.legend()
plt.show() 
plt.plot(0.01*np.arange(len(state_inf[:,11])),state_inf[:,11],label='z')
plt.plot(0.01*np.arange(len(state_inf[:,10])),state_inf[:,10],label='y')
plt.plot(0.01*np.arange(len(state_inf[:,9])),state_inf[:,9],label='x')

plt.ylabel('position/m')
plt.xlabel('time/s')
plt.legend()
plt.show()
plt.plot(0.01*np.arange(len(state_inf[:,6])),state_inf[:,6],label='roll')
plt.ylabel('angle/rad')
plt.xlabel('time/s')
plt.legend()
plt.show()

plt.plot(0.01*np.arange(len(state_inf[:,7])),state_inf[:,7],label='pitch')
plt.ylabel('angle/rad')
plt.xlabel('time/s')
plt.legend()
plt.show()
plt.plot(0.01*np.arange(len(state_inf[:,8])),state_inf[:,8],label='yaw')
plt.ylabel('angle/rad')
plt.xlabel('time/s')
plt.legend()
plt.show()