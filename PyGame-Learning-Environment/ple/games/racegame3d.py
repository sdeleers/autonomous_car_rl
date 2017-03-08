import pygame
from ple.games.utils import percent_round_int
from ple.games.base.pygamewrapper import PyGameWrapper
from pygame.constants import K_a, K_s, K_d
from ple.games import util
import numpy as np
import ple.games.util
import obstacles3d
import race3d
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import h5py
import cPickle
import polygon
import time
import fnmatch
import os
import re
import copy
from scipy.misc import imsave, imrotate

class Car(object):
    def __init__(self, radius, speed, max_speed, pos, car_heading, width, length, height):
        self.radius = radius
        self.speed = speed
        self.max_speed = max_speed
        self.pos = np.array([float(p) for p in pos])
        self.accelerate_factor = 0.0025
        self.decelerate_factor = 0.005
        self.friction_track = 0.999
        self.friction_off_track = 0.98
        self.car_heading = np.deg2rad(car_heading)
        self.max_steer_angle = 90
        self.steer_angle = np.deg2rad(0)
        self.steer_angle_increment = np.deg2rad(2)
        self.width = width
        self.length = length
        self.height = height
        self.wheelbase = length*0.75

    def turn_left(self, env_width):
        if self.speed != 0:
            self.max_steer_angle_for_vel = np.deg2rad(np.min([self.max_steer_angle, 0.1/(self.speed/env_width)**2]))
        else:
            self.max_steer_angle_for_vel = np.deg2rad(self.max_steer_angle)
        self.steer_angle = self.max_steer_angle_for_vel
        # self.steer_angle = np.min([self.steer_angle + self.steer_angle_increment, self.max_steer_angle_for_vel])

    def turn_right(self, env_width):
        if self.speed != 0:
            self.max_steer_angle_for_vel = -np.deg2rad(np.min([self.max_steer_angle, 0.1/(self.speed/env_width)**2]))
        else:
            self.max_steer_angle_for_vel = -np.deg2rad(self.max_steer_angle)
        self.steer_angle = self.max_steer_angle_for_vel
        # self.steer_angle = np.max([self.steer_angle - self.steer_angle_increment, self.max_steer_angle_for_vel])

    def accelerate(self, env_width):
        self.speed = np.min([self.speed + env_width*self.accelerate_factor, self.max_speed])

    def decelerate(self, env_width):
        self.speed = np.max([0, self.speed - env_width*self.decelerate_factor])

    def collision(self, collided_car):
        self.speed *= 0.5
        collided_car.speed = np.min([collided_car.speed + 0.5*self.speed, collided_car.max_speed])

    # Bicycle model
    def update_position(self, dt, off_track):
        if off_track:
            self.speed = np.min([0.1*self.max_speed, self.speed])
        friction = 1

        # friction = self.friction_off_track if off_track else self.friction_track
        self.speed *= self.friction_track

        vel_car = np.array([np.cos(self.car_heading), np.sin(self.car_heading)])
        vel_front_wheel = np.array([np.cos(self.car_heading+self.steer_angle), np.sin(self.car_heading+self.steer_angle)])
        vel_back_wheel = vel_car
        front_wheel_pos = self.pos + self.wheelbase/2*vel_car + dt*self.speed*vel_front_wheel
        back_wheel_pos = self.pos - self.wheelbase/2*vel_car + dt*self.speed*vel_back_wheel

        self.steer_angle = 0

        self.pos = (front_wheel_pos+back_wheel_pos)/2
        self.car_heading = np.arctan2(front_wheel_pos[1]-back_wheel_pos[1], front_wheel_pos[0]-back_wheel_pos[0])

class RaceGame3d(PyGameWrapper):
    def __init__(self, no_cars=1, width=640, height=640):
        # actions = {
        #     "left": K_a,
        #     "right": K_d,
        #     "straight": K_z
        # }
        actions = {
            "left": K_a,
            "right": K_d,
            "accelerate": K_w,
            "decelerate": K_s,
            "straight": K_z
        }
        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.no_cars = no_cars
        self.cars = list()

        self.car_radius = percent_round_int(self.width, 0.005)
        self.car_speed = 0.0*self.width
        self.max_car_speed = 0.25*self.width
        self.car_width = float(self.width)/150
        self.car_length = 1.8*self.car_width
        self.car_height = self.car_width

        self.score_sum = 0.0
        self.no_crashes = 0

        self.reward_track_segment = 10
        self.reward_tick = 0 # disfavor standing still
        self.reward_action = 0
        self.reward_off_track = -15
        self.reward_wrong_direction = -50
        self.reward_crash = -500
        self.reward_car_passed = 500
        self.car_max_off_track_count = 50

        self.camera_viewpoint = 'top'

        # Set agent
        self.set_most_recent_agent()

        self.tracks = race3d.generate_tracks(self.width * 30 / 600, self.width, self.height)
        self.tracks_double = race3d.generate_tracks(1280 * 30 / 600, 1280, 1280)

        # Distance sensors
        # self.sensor_angles = np.array([-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90])
        # self.sensor_angles = np.array([-90, -75, -60, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90])
        self.sensor_angles = np.array([])
        self.no_distance_sensors = self.sensor_angles.shape[0]

        self.timesteps_counter = 0
        self.track_reset_counter = 0

        self.get_straight_segments()

    # def get_straight_segments(self):
    #     # Make sure we start on a straight segment
    #     straight_segments = list()
    #     track = self.tracks[0]
    #     for i in range(track.length):
    #         segment = i
    #         grad_initial = track.gradients[segment]
    #         for k in range(1, 4):
    #             grad = track.gradients[(segment + k) % track.length]
    #             if abs(util.angle_between(grad_initial, grad)) > 5:
    #                 break
    #             elif (k == 3):
    #                 straight_segments.append(segment)
    #     print straight_segments

    # Override setup from pygamewrapper.py
    def _setup(self):
        pygame.display.init()
        pygame.font.init()
        print self.getScreenDims()
        self.screen = pygame.display.set_mode(self.getScreenDims(), DOUBLEBUF | OPENGLBLIT | OPENGL)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(120, 1, 0.1, 1500.0)
        self.clock = pygame.time.Clock()

    def set_agent(self, filename):
        h5f = h5py.File(filename, 'r')
        agent = cPickle.loads(h5f['agent_last_snapshot'].value)
        self.agent = agent
        print "agent set from: " + filename

    def set_most_recent_agent(self):
        agent = 1
        # self.set_agent('/tmp/bla.a')
        # def tryint(s):
        #     try:
        #         return int(s)
        #     except:
        #         return s
        #
        # def alphanum_key(s):
        #     """ Turn a string into a list of string and number chunks.
        #         "z23a" -> ["z", 23, "a"]
        #     """
        #     return [tryint(c) for c in re.split('([0-9]+)', s)]
        #
        # agent_dir = "/tmp/"
        # files = list()
        # for file in os.listdir(agent_dir):
        #     if fnmatch.fnmatch(file, 'h5_*.a'):
        #         files.append(file)
        #
        # files.sort(key=alphanum_key)
        #
        # self.set_agent(agent_dir + files[-1])

    def init(self):
        # Update track and initial car positions
        if (self.track_reset_counter == 0) or (self.track_reset_counter > 50):
            # print "Track reset"
            self.track_reset_counter = 1
            self.current_track_no = np.random.randint(0, len(self.tracks))
            self.current_track_no = 0
            self.track = self.tracks[self.current_track_no]
            self.track_double = self.tracks_double[self.current_track_no]

            self.initial_track_segments = np.zeros(self.no_cars, dtype=int)
            self.initial_track_segments[0] = np.random.randint(0, self.track.length)
            for i in range(1, self.no_cars):
                self.initial_track_segments[i] = (self.initial_track_segments[i - 1] + np.random.randint(20,30)) % self.track.length

            # # Make sure we start on a straight segment
            # for i in range(1, self.no_cars):
            #     straight_initial_segment_found = False
            #     while not straight_initial_segment_found:
            #         self.initial_track_segments = np.zeros(self.no_cars, dtype=int)
            #         self.initial_track_segments[i] = np.random.randint(0, self.track.length)
            #         grad_initial = self.track.gradients[self.initial_track_segments[i]]
            #         for k in range(1, 4):
            #             grad = self.track.gradients[(self.initial_track_segments[0] + k) % self.track.length]
            #             if abs(util.angle_between(grad_initial, grad)) > 5:
            #                 break
            #             elif (k == 3):
            #                 straight_initial_segment_found = True
            #
            # # Initial segments for other cars
            #
            #         self.initial_track_segments[i] = (self.initial_track_segments[i - 1] + np.random.randint(20, 30)) % self.track.length

            # self.initial_car_perturbations = np.random.uniform(-1, 1, self.no_cars)
            # print self.initial_car_perturbations
            # print self.initial_track_segments
            # print self.current_track_no
        # self.track = race3d.generate_circular_track(self.width * 30 / 600, self.width, self.height)


        # Update agent
        if self.timesteps_counter > 20*5000:
            self.set_most_recent_agent()
            self.timesteps_counter = 0

        self.score_sum = 0.0
        self.no_crashes = 0

        # current image state and 3 previous ones
        self.img_buffers = list()
        self.img_buffer_size = 1
        for i in range(self.no_cars):
            self.img_buffers.append(np.zeros((self.width/10, self.height/10, self.img_buffer_size)))

        # Distance sensors
        self.distance_sensors_buffer_size = 4
        self.distance_sensors_buffers = list()
        for i in range(self.no_cars):
            self.distance_sensors_buffers.append(100*np.ones((self.distance_sensors_buffer_size, self.no_distance_sensors), dtype=int))

        self._reset_cars()


    def getScreenRGB(self):
        if self.camera_viewpoint == 'car':
            self.drawCameraForCar(0)
        elif self.camera_viewpoint == 'top':
            self.drawTopViewpoint()
        else:
            raise ValueError('No proper camera viewpoint was set.')

        img = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_INT)
        return img.astype(np.uint8)
        # return np.rot90(np.fliplr(img.astype(np.uint8)))

    def _next_track(self):
        self.current_track_no = np.random.randint(0, len(self.tracks))
        self.track = self.tracks[self.current_track_no]

    def _reset_cars(self):
        self.car_off_track_counts = np.zeros(self.no_cars, dtype=int)
        self.wrong_direction_counter = np.zeros(self.no_cars)

        self.cars = list()

        self.current_track_segments = copy.copy(self.initial_track_segments)
        # self.current_track_segments[0] = self.initial_track_segment
        # self.current_track_segments[0] = np.random.randint(0, self.track.length)

        for i in range(self.no_cars):
            # if i != 0:
            #     # self.current_track_segments[i] = self.current_track_segments[i - 1] + 4
            #     self.current_track_segments[i] = (self.current_track_segments[i-1] + np.random.randint(20, 30)) % self.track.length
            initial_track_segment = self.initial_track_segments[i]
            grad_init = self.track.gradients[initial_track_segment]
            car_heading_init = util.angle_between([1, 0], grad_init)
            if i == 0:
                pos_init = self.track.positions[initial_track_segment]
            else:
                # Random position at segment, not only in center of the track
                perp_grad_init = self.track.perp_gradients[initial_track_segment]
                pos_init = self.track.positions[initial_track_segment]
                # pos_init = self.track.positions[initial_track_segment]+0.4*self.track.width*self.initial_car_perturbations[i]*perp_grad_init

            if i == 0:
                car = Car(self.car_radius, self.car_speed, self.max_car_speed, pos_init, car_heading_init, self.car_width, self.car_length, self.car_height)
            else:
                car = Car(self.car_radius, self.car_speed, self.max_car_speed*0.2, pos_init, car_heading_init,
                          self.car_width, self.car_length, self.car_height)
            self.cars.append(car)

        self.absolute_track_segments = copy.copy(self.initial_track_segments)
        for i in range(1, len(self.absolute_track_segments)):
            if self.absolute_track_segments[i] < self.absolute_track_segments[i-1]:
                self.absolute_track_segments[i] += self.track.length


    def _reset_car(self, car_no):
        # # Find the initial segment as the middle between the 2 segments (at which there are cars) that are farthest
        # # from each other. The car is therefore initialized in a zone with the least amount of cars.
        # sorted_segments = np.sort(self.current_track_segments)
        # segment_diffs = np.diff(sorted_segments)
        # max_segment_diff = np.max(segment_diffs)
        # last_segment_diff = self.track.length - sorted_segments[-1] + sorted_segments[0]
        # if last_segment_diff > max_segment_diff:
        #     # initial_segment = sorted_segments[-1] + int(last_segment_diff/2)
        #     initial_segment = sorted_segments[-1] + np.random.randint(20, 30)
        # else:
        #     # initial_segment = sorted_segments[np.argmax(segment_diffs)] + int(max_segment_diff/2)
        #     initial_segment = sorted_segments[np.argmax(segment_diffs)] + np.random.randint(20, 30)

        segment_diff = 25
        self.absolute_track_segments[car_no] = np.max(self.absolute_track_segments) + segment_diff
        self.current_track_segments[car_no] = self.absolute_track_segments[car_no] % self.track.length
        self.car_off_track_counts[car_no] = 0

        if car_no == 0:
            max_car_speed = self.max_car_speed
            car_speed = self.car_speed
        else:
            max_car_speed = self.max_car_speed*0.2
            car_speed = self.car_speed*0.2

        pos_init = self.track.positions[self.current_track_segments[car_no] % self.track.length]
        grad_init = self.track.gradients[self.current_track_segments[car_no] % self.track.length]
        car_heading_init = util.angle_between([1, 0], grad_init)
        car = Car(self.car_radius, car_speed, max_car_speed, pos_init, car_heading_init, self.car_width,
                  self.car_length, self.car_height)
        self.cars[car_no] = car

    def getScore(self):
        return self.score_sum

    def game_over(self):
        return self.no_crashes == 1

    # def drawCameraForCarAtAngle(self, car_no, angle):
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     obstacles3d.set_car_viewpoint(self.cars[car_no].pos, self.cars[car_no].car_heading+np.deg2rad(angle), self.cars[car_no].height)
    #     race3d.draw_track(self.track)
    #     for i in range(0, self.no_cars):
    #         if i == car_no:
    #             continue
    #         obstacles3d.draw_car(self.cars[i].pos, self.cars[i].car_heading,
    #                              self.cars[i].width, self.cars[i].length, self.car_height, (1.0, 1.0, 1.0))

    def drawCameraForCar(self, car_no):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        obstacles3d.set_car_viewpoint(self.cars[car_no].pos, self.cars[car_no].car_heading, self.cars[car_no].height, self.cars[car_no].length)
        race3d.draw_track(self.track)
        for i in range(0, self.no_cars):
            if i == car_no:
                continue
            obstacles3d.draw_car(self.cars[i].pos, self.cars[i].car_heading,
                                 self.cars[i].width, self.cars[i].length, self.car_height, (1.0, 1.0, 1.0))

    # # Double resolution
    def drawTopViewpoint(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        obstacles3d.set_top_viewpoint(self.width, self.height)
        # race3d.draw_track(self.track)
        race3d.draw_track(self.track_double)

        for i in range(0, self.no_cars):
            if i == 0:
                color = (1.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            obstacles3d.draw_car(2*self.cars[i].pos, self.cars[i].car_heading,
                                 2*self.cars[i].width, 2*self.cars[i].length, 0, color)

    # def drawTopViewpoint(self):
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     obstacles3d.set_top_viewpoint(self.width, self.height)
    #     race3d.draw_track(self.track)
    #
    #     for i in range(0, self.no_cars):
    #         if i == 0:
    #             color = (1.0, 0.0, 0.0)
    #         else:
    #             color = (1.0, 1.0, 1.0)
    #         obstacles3d.draw_car(self.cars[i].pos, self.cars[i].car_heading,
    #                              self.cars[i].width, self.cars[i].length, 0, color)

    def getGameStateForCar(self, car_no):
        # skipped_buffer = (self.img_buffers[car_no])[:, :, (0, 1, 2, 3)].ravel()
        skipped_buffer = (self.img_buffers[car_no])[:, :, 0].ravel()
        # return np.concatenate((skipped_buffer, [self.cars[car_no].speed]))
        # return np.concatenate((self.img_buffers[car_no].ravel(), [self.cars[car_no].speed]))
        return np.concatenate((self.img_buffers[car_no].ravel(), [self.cars[car_no].speed], self.distance_sensors_buffers[car_no].ravel()))

    def getGameState(self):
        # Change the viewpoint to the car if viewpoint is top, since we always use the car viewpoint as gamestate.
        return self.getGameStateForCar(0)

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                car = self.cars[0]
                key = event.key
                if key == self.actions['straight']:
                    return
                elif key == self.actions['left']:
                    self.score_sum += self.reward_action
                    car.turn_left(self.width)
                elif key == self.actions['right']:
                    self.score_sum += self.reward_action
                    car.turn_right(self.width)
                elif key == self.actions['accelerate']:
                    self.score_sum += self.reward_action
                    car.accelerate(self.width)
                elif key == self.actions['decelerate']:
                    self.score_sum += self.reward_action
                    car.decelerate(self.width)

    # Process other car events (car_no should not be 0, car 0 is updated in _handle_player_events)
    def _handle_other_car_events(self, car_no, action):
        # car = self.cars[car_no]
        if action == 0:
            # left
            self.cars[car_no].turn_left(self.width)
        elif action == 1:
            # right
            self.cars[car_no].turn_right(self.width)
        elif action == 2:
            # decelerate
            self.cars[car_no].decelerate(self.width)
        elif action == 3:
            # accelerate
            self.cars[car_no].accelerate(self.width)
        elif action == 4:
            # straight
            return

    def step(self, dt):
        dt /= 1000.0

        self.score_sum += self.reward_tick

        for i in range(1, len(self.cars)):
            absolute_track_segments_diff = self.absolute_track_segments[0]-self.absolute_track_segments
            car_reset_indices = np.where(absolute_track_segments_diff > 0)[0]
            for index in car_reset_indices:
                self._reset_car(index)
                self.score_sum += self.reward_car_passed

        if self.timesteps_counter % 10 == 0:
            a = 2

        # # Get images from 3 angles
        # for car_no in range(1):
        #     angles = [-90, 0, 90]
        #     for i in range(3):
        #         self.drawCameraForCarAtAngle(car_no, angles[i])
        #         img = glReadPixels(0, 0, self.width, self.height, GL_RED, GL_BYTE)
        #         (self.img_buffers[car_no])[:, :, i] = img[::10, ::10]

        # Shift sensor distances buffer
        for car_no in range(1):
        # for car_no in range(self.no_cars):
            for i in reversed(range(1, self.distance_sensors_buffer_size)):
                (self.distance_sensors_buffers[car_no])[i, :] = (self.distance_sensors_buffers[car_no])[i - 1, :]
            (self.distance_sensors_buffers[car_no])[0, :] = race3d.sensor_distances_between_cars(self.cars, car_no, self.sensor_angles, self.car_length, self.car_width)

        # Handle keypresses for car 0
        self._handle_player_events()

        # for i in range(1, self.no_cars):
        #     ob = self.getGameStateForCar(i).ravel()
        #     ob = self.agent.obfilt(ob)
        #     action, agentinfo = self.agent.act(ob)
        #     self._handle_other_car_events(i, action)

        for i in range(self.no_cars):
            car = self.cars[i]
            # Don't waste computing time for non-moving obstacles
            if car.speed != 0:
                car.update_position(dt, self.car_off_track_counts[i] != 0)

                if race3d.circular_car_off_track(car.pos, self.track):
                    self.car_off_track_counts[i] += 1
                    self.wrong_direction_counter[i] = 0
                    if i == 0:
                        self.score_sum += self.reward_off_track
                    if self.car_off_track_counts[i] > self.car_max_off_track_count:
                        if i == 0:
                            # Only reset when car0 crashed
                            self.no_crashes += 1
                            # return
                        else:
                            # Only reset car that crashed if car0 did not crash
                            self._reset_car(i)
                # If on track
                else:
                    new_track_segment = race3d.get_current_track_segment(car.pos, self.track)
                    # If previous step on track
                    if self.car_off_track_counts[i] == 0:
                        segment_diff = new_track_segment - self.current_track_segments[i]
                        score_sum_change = 0
                        # if track finished
                        if abs(segment_diff) > self.track.length/2:
                            segment_diff = (self.track.length - abs(segment_diff))
                        score_sum_change = self.reward_track_segment*segment_diff
                        self.absolute_track_segments[i] += segment_diff

                        if i == 0:
                            self.score_sum += score_sum_change
                        # Check whether car going in correct direction. If car0 not going in correct direction stop simulation to avoid incorrect learning
                        # (since following the track correctly but in wrong direction will discourage following the track)
                        if score_sum_change < 0:
                            self.wrong_direction_counter[i] += 1
                            if self.wrong_direction_counter[i] > 5:
                                if i == 0:
                                    self.score_sum += self.reward_wrong_direction
                                    self.no_crashes += 1
                                else:
                                    self._reset_car(i)
                        elif score_sum_change > 0:
                            self.wrong_direction_counter[i] = 0
                    # First step back on track
                    else:
                        if new_track_segment >= self.current_track_segments[i]:
                            number_of_segments_skipped = abs(new_track_segment-self.current_track_segments[i])
                        else:
                            number_of_segments_skipped = abs(self.track.length-self.current_track_segments[i]+new_track_segment)
                        self.absolute_track_segments[i] += number_of_segments_skipped
                        if i == 0:
                            self.score_sum += self.reward_off_track*number_of_segments_skipped

                    self.current_track_segments[i] = new_track_segment
                    self.car_off_track_counts[i] = 0

        # Car collisions
        for i in range(0, self.no_cars):
            car_i_polygon = race3d.carToPolygon(self.cars[i], self.car_length, self.car_width)
            for j in range(i+1, self.no_cars):
                car_j_polygon = race3d.carToPolygon(self.cars[j], self.car_length, self.car_width)
                if car_i_polygon.collidepoly(car_j_polygon):
                    if (i == 0):
                        self.score_sum += self.reward_crash
                        self.no_crashes += 1
                    if self.current_track_segments[i] > self.current_track_segments[j]:
                        # Don't give our agent an incentive to crash in the car behind him to gain speed
                        if i != 0:
                            self.cars[j].collision(self.cars[i])
                    elif self.current_track_segments[i] < self.current_track_segments[j]:
                        self.cars[i].collision(self.cars[j])
                    # self.no_crashes += 1

        # Shift image buffers
        for car_no in range(self.no_cars):
        # for car_no in range(self.no_cars):
            self.drawCameraForCar(car_no)
            # Shift buffer
            for i in reversed(range(1, self.img_buffer_size)):
                (self.img_buffers[car_no])[:, :, i] = (self.img_buffers[car_no])[:, :, i - 1]
            # Add new image to buffer
            img = glReadPixels(0, 0, self.width, self.height, GL_RED, GL_BYTE)
            (self.img_buffers[car_no])[:, :, 0] = img[::10, ::10]

        # Draw everything on screen
        if self.camera_viewpoint == 'car':
            self.drawCameraForCar(0)
        elif self.camera_viewpoint == 'top':
            self.drawTopViewpoint()
        else:
            raise ValueError('No proper camera viewpoint was set.')

        # Record video
        img_top = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_INT).astype(np.uint8)

        img_combined = np.zeros((self.width, self.width*16/9+1, 3), dtype=np.uint8)
        # img_combined[:, :self.width, :] = imrotate(np.fliplr(conc3), 180)

        # Distance readings
        # no_sensors = len(self.sensor_angles)

        # # distances = self.distance_sensors_buffers[0]
        distances = np.zeros((4, 9))
        distances[0, :] = np.random.randint(0, 100, 9)
        distances[1, :] = np.random.randint(0, 100, 9)
        distances[2, :] = np.random.randint(0, 100, 9)
        distances[3, :] = np.random.randint(0, 100, 9)
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # obstacles3d.draw_distance_sensor_readings(distances)
        # img_distance_sensors = glReadPixels(0, 0, self.width/2, self.height/2, GL_RGB, GL_UNSIGNED_INT).astype(np.uint8)
        # img_combined[0:self.width/2,996:self.width/2+996, :] = np.fliplr(np.flipud(img_distance_sensors))

        # p0 = [self.width/2, self.width]
        # for i in range(distances.shape[0]):
        #     for j in range(distances.shape[1]):
        #         dist_reading = obstacles3d.get_distance_reading(distances[i, j], 16)
        #         img_combined[p0[0]+j*(16+8):p0[0]+j*(16+8)+16, p0[1]+i*(100+10):p0[1]+i*(100+10)+100, :] = np.swapaxes(dist_reading, 0, 1)


        # skipped_buffer = (self.img_buffers[0])[:, :, (0, 4, 8, 12)]
        # img_speed_reading = obstacles3d.get_speed_reading(self.cars[0].speed)
        # tmp = (self.width*7/9-self.width/2+1) / 2
        # img_combined[self.width/2+50:self.width/2+50+32, 150+self.width+tmp:150+self.width+tmp+480, :] = np.swapaxes(img_speed_reading, 0, 1)
        #

        # skipped_buffer = (self.img_buffers[0])[:, :, 0]


        # img_tmp1 = imrotate(np.fliplr(img_tmp1), 180)
        # imsave('/Users/simon/MachineLearning/modular_rl/viewpoints_combined/tmp1.png', img_tmp1.astype(np.uint8))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.drawCameraForCar(0)
        img = glReadPixels(0, 0, self.width, self.height, GL_RED, GL_BYTE)
        img_car = imrotate(np.repeat(np.expand_dims(img.astype(np.uint8), 2), 3, 2), 180)[::2, ::2]
        tmp = (self.width*7/9-self.width/2+1)/2
        img_combined[20:20+self.width/2, self.width+tmp:self.width+tmp+self.width/2, :] = np.fliplr(img_car)

        img = glReadPixels(0, 0, self.width, self.height, GL_RED, GL_BYTE)

        # imsave('viewpoints_combined/%05d.png' % (self.timesteps_counter), img_combined)
        imsave('/Users/simon/MachineLearning/modular_rl/viewpoints_combined/%05d.png' % (self.timesteps_counter), img_combined)

        # Update counters
        self.track_reset_counter += 1
        self.timesteps_counter += 1


if __name__ == "__main__":
    pygame.init()

    game_width = 640
    game_height = 640

    game = RaceGame3d(1, width=game_width, height=game_height)
    pygame.display.set_mode(game.getScreenDims(), DOUBLEBUF | OPENGL)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(120, 1, 0.1, 1500.0)

    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    while True:
        dt = game.clock.tick_busy_loop(60)
        game.step(dt)
        pygame.display.flip()
