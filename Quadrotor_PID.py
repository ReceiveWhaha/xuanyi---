# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 22:54:37 2021

@author: 14791
"""


'''##log:8.14:
    ##已经完成了巡航的路径点生成，转弯问题仍然没有解决，问题在于飞控
'''

import numpy as np

#import time
import cv2

from Quadrotor_vrep import SpaceRobot3link
import matplotlib.pyplot as plt

# Environment

env = SpaceRobot3link()
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# env.seed(args.seed)
# Agent

# Memory

total_numsteps = 0
updates = 0
totalr = []
step = 150


start_z_= 5
start_x_= 2
start_y_= 0
# 起始位置

target_z_ = 5
target_x_ = 4
target_y_ = 3
# 最终目标点

#steplen = 1#步长，可以控制速度
captime = 60#图片采样间隔



kp_xx = 0.5
kd_xx = 0.8
kp_yy = -0.5
kd_yy = -0.8

target_yaw = 0
target_roll = 0
target_pitch = 0


Kp_y=100
Kd_y=20
Kp_r=100
Kd_r=200
Kp_p=500
Kd_p=100
Kp_z=300
kd_z=50
Kp_joint=-0.025
Kd_joint=-0.005
flag=0#counter,to name the pic

# def tarlistGet(tarX,tarY)
# return tarlist
# return tarnum
# TARLIST, TARNUM = tarlistGet(target_x_ , target_y_ )



#def tarnumGetY(staY,tarY):#纵向节点数获取
    #if abs(tarY-staY)%1 == 0:
        #tarnumY = abs(tarY-staY)/1
    #else:
        #tarnumY = abs(tarY-staY)/1 + 1
    #return int(tarnumY)



#def tarlistGetY(tarX, staY, tarY, tarlist, tarnum):
    
    #for i in range(tarnum):                    
        #tarlist[i][0] = tarX
        #tarlist[i][1] = staY+ (i+1)*1*((tarY-staY)/abs(tarY-staY))
        #if i == tarnum-1:
            #tarlist[i][0] = tarX
            #tarlist[i][1] = tarY
    #return tarlist



def tar_numGet(staX, tarX):#节点目标数获取
    tarnum_ = 2*(tarX - staX) + 2
    return  tarnum_


def tar_listget(staX, staY, tarY, tar_list, tar_num):#节点目标获取
    for i in range (tar_num):
        tar_list[i][1] = staX + int(i/2)
        if i%4 == 1 or i%4 == 2:
            tar_list[i][0] = tarY
        else:
            tar_list[i][0] = staY
    return tar_list      
 

           
def grouplistGet(sy, ty, x, grouplist):#部分路径计算
    for i in range (abs(ty - sy)+2):
        if i == abs(ty - sy)+1:
            grouplist[i][0] = ty
            grouplist[i][1] = x
        else:
            if ty > sy:
               grouplist[i][0] = sy + i
            else :
               grouplist[i][0] = sy - i
            grouplist[i][1] = x
    return grouplist
            
def GlistGet(tar_list, G, gnum):#部分路径生成
    for i in range(gnum):
        sy = tar_list[2*i][0]
        ty = tar_list[2*i + 1][0]
        x = tar_list[2*i][1]
        G[i] = grouplistGet(sy, ty, x, G[i])
    return G


def tarnumGet(staX, tarX, staY, tarY):#总路径点数计算
    tarnum = (abs(tarX - staX) + 1) * (abs(tarY - staY) +1) + abs(tarX - staX) + 1 
    return tarnum   
         
         

           
def tarlistGet(G, gnum, tarlist):  
    for i in range(gnum):
        tarlist = tarlist + G[i]
    return tarlist



TAR_NUM = tar_numGet(start_x_, target_x_)
TAR_LIST = [[0 for col in range(2)] for row in range(TAR_NUM)]
TAR_LIST = tar_listget(start_x_, start_y_, target_y_, TAR_LIST, TAR_NUM)

gnum = abs(target_x_ - start_x_) + 1
glenth = abs(target_y_  - start_y_) + 2

G = [[[0 for col in range(2)] for row in range(glenth)] for x in range(gnum)]#生成存放分段路径列表的列表
G = GlistGet(TAR_LIST, G, gnum)

TARNUM = tarnumGet(start_x_, target_x_ , start_y_ , target_y_)
TARLIST = []
TARLIST = tarlistGet(G, gnum, TARLIST)
print(TARLIST)

for i_episode in range(1):
    state, visual = env.reset()
    episode_reward = 0
    episode_steps = 0
    done = False
    r = []
    control_information = np.zeros([step, 4])
    state_inf = np.zeros([step, 12])

    for k in range(TARNUM):
        target_z = target_z_
        target_x = TARLIST[k][1]
        #target_x = target_x_
        target_y = TARLIST[k][0]
        print(target_x, target_y)

        for i_epoch in range(step):
            #完全随机动作序列
            z=state[5]

            v_z=state[8]
            p_z=Kp_z*(target_z-z)-kd_z*v_z
            
            target_roll=kp_yy*(target_y-state[4])-kd_yy*state[7]
            target_pitch=kp_xx*(target_x-state[3])-kd_xx*state[6]
            
            #target_pitch=0
            #target_roll=0
            #target_yaw=0
            output_yaw=Kp_y*(target_yaw-state[2])+Kd_y*(-state[11])
            output_roll=Kp_r*(target_roll-state[0])+Kd_r*(-state[9])
            output_pitch=Kp_p*(target_pitch-state[1])+Kd_p*(-state[10])
            
            
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
            
            
            
            output_t_r=p_z-output_roll
            output_t_l=p_z+output_roll
            
            output_1=p_z-output_pitch-output_yaw
            output_2=p_z-output_roll+output_yaw
            output_3=p_z+output_pitch-output_yaw
            output_4=p_z+output_roll+output_yaw
            
            action =np.array([output_1,output_2,output_3,output_4])

            #连续正弦动作序列
            
            next_state,visual, reward, done, _ = env.step(action) # Step
            plt.imshow(visual)
            plt.ion()
            plt.show();
            if flag%captime ==0:#speed control
                flag1=flag/captime 
                flag0='%d'%flag1
                img_Name = "../output/" + flag0 + ".png"
                b,g,r = cv2.split(visual)#拆分通道
                visual1 = cv2.merge([r,g,b])#合并通道
                cv2.imwrite(img_Name,visual1)
            control_information[i_epoch]=action
            state_inf[i_epoch]=state
            state=next_state
            flag=flag+1
        #time.sleep(2)  

env.close()

#plt.plot(0.01*np.arange(len(state_inf[:,5])),state_inf[:,5],label='z')
plt.plot(0.01*np.arange(len(state_inf[:,8])),state_inf[:,8],label='vz')
plt.plot(0.01*np.arange(len(state_inf[:,7])),state_inf[:,7],label='vy')
plt.plot(0.01*np.arange(len(state_inf[:,6])),state_inf[:,6],label='vx')
plt.ylabel('velocity/m')
plt.xlabel('time/s')
#plt.plot(np.arange(len(state_inf[:,2])),state_inf[:,11],label='z')
plt.legend()
plt.show() 

plt.plot(0.01*np.arange(len(state_inf[:,3])),state_inf[:,3],label='x')
plt.plot(0.01*np.arange(len(state_inf[:,4])),state_inf[:,4],label='y')
plt.plot(0.01*np.arange(len(state_inf[:,5])),state_inf[:,5],label='z')

plt.ylabel('position/m')
plt.xlabel('time/s')
plt.legend()
plt.show()

plt.plot(0.01*np.arange(len(state_inf[:,6])),state_inf[:,0],label='roll')
plt.plot(0.01*np.arange(len(state_inf[:,7])),state_inf[:,1],label='pitch')
plt.plot(0.01*np.arange(len(state_inf[:,8])),state_inf[:,2],label='yaw')
plt.ylabel('angle/rad')
plt.xlabel('time/s')
plt.legend()
plt.show()