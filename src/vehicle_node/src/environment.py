#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import copy
import bisect
import matplotlib.cm as cm
import matplotlib.animation as animation

from IPython.display import HTML
from utils import *
from agent import agent
from kalman_filter import *
from imm_filter import *


import tf
import rospkg
import rospy

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray


class Environments(object):
    def __init__(self, course_idx, dt=0.1, min_num_agent=8, num_kalman_data=10, t_pred=2.0):

        self.spawn_id = 0
        self.vehicles = {}
        self.int_pt_list = {}
        self.sensor_info_dict = {}
        self.kalman_filter_dict = {}
        self.kf_pred_dict = {}
        self.model_pred_dict = {}
        self.min_num_agent = min_num_agent
        self.dt = dt
        self.course_idx = course_idx
        self.num_kalman_data = num_kalman_data
        self.t_pred = t_pred
        self.check_intersection_distance = 20
        self.curvature_speed_factor = 0.01
        self.k_p = 5.0
        self.v_max = 8.0

        self.initialize()

    def initialize(self, init_num=6):

        self.pause = False
        filepath = rospy.get_param("file_path")
        Filelist = glob.glob(filepath+"/*info.pickle")

        file = Filelist[0]

        with open(file, "rb") as f:
            Data = pickle.load(f)

        self.map_pt = Data["Map"]
        self.connectivity = Data["AS"]

        for i in range(init_num):
            if i == 0:
                CourseList = [[4, 1, 18], [4, 2, 25], [4, 0, 11]]
                self.spawn_agent(
                    target_path=CourseList[self.course_idx], init_v=0)
            else:
                self.spawn_agent()

    def spawn_agent(self, target_path=[], init_v=None):

        is_occupied = True

        if target_path:

            spawn_cand_lane = target_path[0]
            is_occupied = False
            s_st = 5

        else:
            spawn_cand_lane = [10, 12, 24, 17, 19]

            s_st = np.random.randint(0, 20)
            max_cnt = 10
            while (is_occupied and max_cnt > 0):

                spawn_lane = np.random.choice(spawn_cand_lane)

                is_occupied = False
                for id_ in self.vehicles.keys():
                    if (self.vehicles[id_].lane_st == spawn_lane) and np.abs(self.vehicles[id_].s - s_st) < 25:
                        is_occupied = True

                max_cnt -= 1

        if is_occupied is False:
            if target_path:
                target_path = target_path

            else:
                target_path = [spawn_lane]
                spawn_lane_cand = np.where(
                    self.connectivity[spawn_lane] == 1)[0]

                while (len(spawn_lane_cand) > 0):
                    spawn_lane = np.random.choice(spawn_lane_cand)
                    target_path.append(spawn_lane)
                    spawn_lane_cand = np.where(
                        self.connectivity[spawn_lane] == 1)[0]

            target_pt = np.concatenate(
                [self.map_pt[lane_id][:-1, :] for lane_id in target_path], axis=0)
            self.int_pt_list[self.spawn_id] = {}

            for key in self.vehicles.keys():
                intersections = find_intersections(
                    target_pt[:, :3], self.vehicles[key].target_pt[:, :3])  # ((x,y), i, j)

                if intersections:
                    self.int_pt_list[self.spawn_id][key] = [
                        (inter, xy[0], xy[1]) for (inter, xy) in intersections]
                    self.int_pt_list[key][self.spawn_id] = [
                        (inter, xy[1], xy[0]) for (inter, xy) in intersections]

            stopline_idx = len(self.map_pt[target_path[0]])-1
            endline_idx = len(
                self.map_pt[target_path[0]])+len(self.map_pt[target_path[1]])-2

            self.vehicles[self.spawn_id] = agent(self.spawn_id, target_path, s_st, target_pt, dt=self.dt, init_v=init_v,
                                                 stoplineidx=stopline_idx, endlineidx=endline_idx)
            self.spawn_id += 1

    def delete_agent(self):

        delete_agent_list = []

        for id_ in self.vehicles.keys():
            if (self.vehicles[id_].target_s[-1]-10) < self.vehicles[id_].s:
                delete_agent_list.append(id_)

        return delete_agent_list

    def run(self):  # 10hz로 동작(main.py의 r = rospy.Rate(10)과 동일)

        for id_ in self.vehicles.keys():
            if id_ == 0:
                sensor_info_local = self.vehicles[id_].get_measure(
                    self.vehicles)
                local_lane_info = self.vehicles[id_].get_local_path()
                # TODO:
                # 아래의 정보들을 활용하여, SDV가 주변 agent와 충돌 없이
                # 교차로를 통과하여 target lane에 가기 위한 종 / 횡 방향 제어기 설계
                # - sensor info [obj id, rel x, rel y, rel h, rel vx, rel vy]
                # - local lane info [x, y, h, R]
                # - SDV info : self.vehicles[id_].~ [x, y, h, v, s, d]
                # - Global map info : self.map_pt / self.connectivity\

                """
                sensor_info: local frame to global frame
                """
                sensor_info_global = []  # [[obj id, rel x, rel y, rel h, rel vx, rel vy], …]
                for i in (sensor_info_local):
                    obj_id, rel_x, rel_y, rel_h, rel_vx, rel_vy = i
                    x, y, h, vx, vy = local_to_global(
                        self.vehicles[id_].x, self.vehicles[id_].y, self.vehicles[id_].h, self.vehicles[id_].v,
                        rel_x, rel_y, rel_h, rel_vx, rel_vy)
                    sensor_info_global.append([obj_id, x, y, h, vx, vy])

                """
                self.num_kalman_data개의 타입 스탬프 만큼의 데이터를 저장
                """
                # self.sensor_info_dict에 각 agent의 정보 저장
                for info in (sensor_info_global):
                    obj_id, x, y, h, vx, vy = info
                    v = (vx**2 + vy**2) ** 0.5
                    # 해당 obj_id가 sensor_info_dict에 이미 존재하는지 확인
                    if obj_id in self.sensor_info_dict:
                        # 이미 존재하면 해당 obj_id의 리스트에 새로운 데이터 추가
                        self.sensor_info_dict[obj_id].append([x, y, h, v])
                    else:
                        # 존재하지 않으면 새로운 키-값 쌍 추가
                        self.sensor_info_dict[obj_id] = [[x, y, h, v]]
                    # obj_id당 최대 self.num_kalman_data개까지만 저장 (나중에 a, yaw_rate 추가한 후 마지막 항은 사용 X)
                    if len(self.sensor_info_dict[obj_id]) > self.num_kalman_data:
                        self.sensor_info_dict[obj_id].pop(0)

                """
                local_lane_info를 global frame으로 변환 후 self.path_0에 저장
                """
                # local_lane_info를 global로 변환 후 self.path_0에 저장, self.check_intersection_distance만큼만 저장
                local_lane_info = self.vehicles[id_].get_local_path()
                distance = 0
                self.path_0 = []
                for i in range(len(local_lane_info)):
                    x, y, h, R = local_lane_info[i]
                    x, y, h, *_ = local_to_global(
                        self.vehicles[id_].x, self.vehicles[id_].y, self.vehicles[id_].h, self.vehicles[id_].v, x, y, h, 0, 0)
                    self.path_0.append((x, y))
                    if i > 0:
                        distance += np.sqrt((self.path_0[i][0] - self.path_0[i-1][0])**2 +
                                            (self.path_0[i][1] - self.path_0[i-1][1])**2)
                    if distance > self.check_intersection_distance:
                        break

            # self.sensor_info_dict에 a와 yaw_rate 추가
            # self.sensor_info_dict[id_]: [[x, y, h, v, a, yaw_rate], ...]
            if id_ > 0 and id_ in self.sensor_info_dict and len(self.sensor_info_dict[id_]) > 1:
                for i in range(len(self.sensor_info_dict[id_])-1):
                    x1, y1, h1, v1, *_ = self.sensor_info_dict[id_][i]
                    x2, y2, h2, v2, *_ = self.sensor_info_dict[id_][i+1]
                    a = (v2 - v1) / self.dt
                    yaw_rate = (h2 - h1) / self.dt
                    yaw_rate = np.arctan2(np.sin(yaw_rate), np.cos(yaw_rate))
                    if len(self.sensor_info_dict[id_][i]) < 6:
                        self.sensor_info_dict[id_][i].append(a)
                        self.sensor_info_dict[id_][i].append(yaw_rate)
                    else:
                        self.sensor_info_dict[id_][i][4] = a
                        self.sensor_info_dict[id_][i][5] = yaw_rate

                """
                IMM Filtering
                """
                num_of_model = 2
                mat_trans = np.array([[0.85, 0.15], [0.15, 0.85]])
                mu = [1.0, 0.0]

                filters = [Extended_KalmanFilter(
                    5, 4), Extended_KalmanFilter(6, 4)]
                models = [CA(), CTRA()]

                Q_list = [[0.1, 0.1, 0.1, 0.1, 0.001],
                          [0.1, 0.1, 0.1, 0.1, 0.001, 0.01]]

                x = [np.array([self.sensor_info_dict[id_][0][0], self.sensor_info_dict[id_][0][1], self.sensor_info_dict[id_][0][3], 0, self.sensor_info_dict[id_][0][2]]),
                     np.array([self.sensor_info_dict[id_][0][0], self.sensor_info_dict[id_][0][1], self.sensor_info_dict[id_][0][3], 0, self.sensor_info_dict[id_][0][2], 0])]

                for i in range(len(filters)):
                    filters[i].F = models[i].step
                    filters[i].H = models[i].H
                    filters[i].JA = models[i].JA
                    filters[i].JH = models[i].JH
                    filters[i].Q = np.diag(Q_list[i])
                    filters[i].R = np.diag([0.1, 0.1, 0.1, 0.1])
                    filters[i].x = x[i]

                IMM = IMM_filter(filters, mu, mat_trans)
                MM = [mu]
                X = [x[1]]
                Traj = []

                for i in range(len(self.sensor_info_dict[id_]) - 1):

                    z = [self.sensor_info_dict[id_][i][0], self.sensor_info_dict[id_][i][1],
                         self.sensor_info_dict[id_][i][3], self.sensor_info_dict[id_][i][2]]

                    IMM.prediction()
                    IMM.merging(z)

                    traj = IMM.predict(self.t_pred)
                    Traj.append(traj)
                    MM.append(IMM.mu.copy())
                    X.append(IMM.x.copy())

                # 해당 agent의 Kalman Filter 결과 저장
                # {obj_id: [x, y, v, a, theta, theta_rate] * (t_pred / self.dt), ...}
                self.kalman_filter_dict[id_] = X
                self.kf_pred_dict[id_] = traj

            """
            Control
            """
            if id_ == 0:
                # 곡률 반지름에 따른 목표 속도 계산
                radius_of_curvature_avg = np.mean(
                    [info[3] for info in local_lane_info])
                v_cur = self.vehicles[id_].v
                v_target = self.curvature_speed_factor * radius_of_curvature_avg

                # 다른 agent의 예측 경로와 SDV의 실제 경로를 비교하여 충돌 여부 확인
                for kf_pred in self.kf_pred_dict.values():
                    path_id_ = []
                    for info in kf_pred:
                        path_id_.append((info[0], info[1]))
                    intersections = find_intersections_with_indices(
                        self.path_0, path_id_)
                    print(intersections)
                    if len(intersections) > 0:
                        print("Collision detected")
                        v_target = 0
                        break
                v_target = np.clip(v_target, 0, self.v_max)
                ax = self.k_p * (v_target - v_cur)
                # SDV의 제어 입력
                self.vehicles[id_].step_manual(ax, steer=0)

            if id_ > 0:
                self.vehicles[id_].step_auto(
                    self.vehicles, self.int_pt_list[id_])

    def respawn(self):
        if len(self.vehicles) < self.min_num_agent:
            self.spawn_agent()


if __name__ == '__main__':

    try:
        f = Environments()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')
