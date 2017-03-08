from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from util import perpendicular

box_edges = ((0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7), (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7))
box_surfaces = ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6))


# Input: list of 2D positions of square centers (obstacles cubes always lie on the x-y-plane)
def draw_obstacles(obstacle_centers, width, height):
    for center in obstacle_centers:
        vertices = list()
        vertices.append((center[0]+width/2, center[1]-width/2, 0))
        vertices.append((center[0]+width/2, center[1]+width/2, 0))
        vertices.append((center[0]-width/2, center[1]+width/2, 0))
        vertices.append((center[0]-width/2, center[1]-width/2, 0))
        vertices.append((center[0]+width/2, center[1]-width/2, height))
        vertices.append((center[0]+width/2, center[1]+width/2, height))
        vertices.append((center[0]-width/2, center[1]-width/2, height))
        vertices.append((center[0]-width/2, center[1]+width/2, height))
        draw_box_with_surfaces(vertices, (1.0, 1.0, 1.0))


def draw_box_with_lines(vertices):
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)
    for edge in box_edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def draw_box_with_surfaces(vertices, color):
    glBegin(GL_QUADS)
    glColor3fv(color)
    for surface in box_surfaces:
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()


def draw_circle(radius, pos):
    glBegin(GL_LINE_LOOP)
    num_segments = 100
    glColor3fv((1.0, 1.0, 1.0))
    for i in range(num_segments):
        theta = 2*np.pi*i/num_segments
        glVertex3fv((radius*np.cos(theta) + pos[0], radius*np.sin(theta) + pos[1], 0))
    glEnd()


def draw_ground(env_width, env_height):
    glBegin(GL_QUADS)
    ground_vertices = ((0, 0, 0), (env_height, 0, 0), (env_height, env_width, 0), (0, env_width, 0))
    # glColor3fv((84.0 / 256, 186.0 / 256, 91.0 / 256))
    glColor3fv((0.0, 0.0, 0.0))
    for vertex in ground_vertices:
        glVertex3fv(vertex)
    glEnd()


def draw_walls(env_width, env_height, wall_height, wall_thickness):
    w = env_width
    h = env_height
    t = wall_thickness
    wall_south = ((-t, 0, 0), (w+t, 0, 0), (w+t, 0, wall_height), (0-t, 0, wall_height), (0-t, 0-t, 0),
                  (w+t, 0-t, 0), (0-t, 0-t, wall_height), (w+t, 0-t, wall_height))
    wall_north = ((0-t, h, 0), (w+t, h, 0), (w+t, h, wall_height), (0-t, h, wall_height), (0-t, h+t, 0),
                  (w+t, h+t, 0), (0-t, h+t, wall_height), (w+t, h+wall_thickness, wall_height))
    wall_east = ((0, h, 0), (0, 0, 0), (0, 0, wall_height), (0, h, wall_height), (0-t, h, 0),
                 (0-t, 0, 0), (0-t, h, wall_height), (0-t, 0, wall_height))
    wall_west = ((w, h, 0), (w, 0, 0), (w, 0, wall_height), (w, h, wall_height), (w+t, h, 0),
                 (w+t, 0, 0), (w+t, h, wall_height), (w+t, 0, wall_height))
    draw_box_with_surfaces(wall_south, (1.0, 1.0, 1.0))
    draw_box_with_surfaces(wall_north, (1.0, 1.0, 1.0))
    draw_box_with_surfaces(wall_east, (1.0, 1.0, 1.0))
    draw_box_with_surfaces(wall_west, (1.0, 1.0, 1.0))


def draw_car_circle(pos, radius):
    draw_circle(radius, pos)


def draw_car(pos, car_heading, width, length, height, color):
    car_vel = np.array([np.cos(car_heading), np.sin(car_heading)])
    car_vel_perp = perpendicular(car_vel)
    pert_vel = length/2*car_vel
    pert_vel_perp = width/2*car_vel_perp
    vertices = list()
    vertices.append((pos[0]+pert_vel[0]-pert_vel_perp[0], pos[1]+pert_vel[1]-pert_vel_perp[1], 0))
    vertices.append((pos[0]+pert_vel[0]+pert_vel_perp[0], pos[1]+pert_vel[1]+pert_vel_perp[1], 0))
    vertices.append((pos[0]-pert_vel[0]+pert_vel_perp[0], pos[1]-pert_vel[1]+pert_vel_perp[1], 0))
    vertices.append((pos[0]-pert_vel[0]-pert_vel_perp[0], pos[1]-pert_vel[1]-pert_vel_perp[1], 0))
    vertices.append((pos[0]+pert_vel[0]-pert_vel_perp[0], pos[1]+pert_vel[1]-pert_vel_perp[1], height))
    vertices.append((pos[0]+pert_vel[0]+pert_vel_perp[0], pos[1]+pert_vel[1]+pert_vel_perp[1], height))
    vertices.append((pos[0]-pert_vel[0]-pert_vel_perp[0], pos[1]-pert_vel[1]-pert_vel_perp[1], height))
    vertices.append((pos[0]-pert_vel[0]+pert_vel_perp[0], pos[1]-pert_vel[1]+pert_vel_perp[1], height))
    draw_box_with_surfaces(vertices, color)


def get_speed_reading(speed):
    width = 32
    max_speed = 160
    speed_reading = np.zeros((3*max_speed, width, 3))
    if speed < max_speed / 6:
        speed_reading[:int(3*speed), :, 2] = 0.4*255
    elif speed < max_speed * 2 / 6:
        speed_reading[:int(3*speed), :, 2] = 0.7*255
    elif speed < max_speed * 3 / 6:
        speed_reading[:int(3*speed), :, 2] = 255.0
    elif speed < max_speed * 4 / 6:
        speed_reading[:int(3*speed), :, 0] = 255.0
    elif speed < max_speed * 5 / 6:
        speed_reading[:int(3*speed), :, 0] = 0.7 * 255
    else:
        speed_reading[:int(3*speed), :, 0] = 0.4*255
    return speed_reading

def get_distance_reading(distance, width):
    max_dist = 100
    distance_reading = np.zeros((max_dist, width, 3))
    if distance < max_dist/6:
        distance_reading[:int(distance), :, 2] = 0.4 * 255
    elif distance < max_dist*2/6:
        distance_reading[:int(distance), :, 2] = 0.7*255
    elif distance < max_dist*3/6:
        distance_reading[:int(distance), :, 2] = 255.0
    elif distance < max_dist*4/6:
        distance_reading[:int(distance), :, 0] = 255.0
    elif distance < max_dist*5/6:
        distance_reading[:int(distance), :, 0] = 0.7 * 255
    else:
        distance_reading[:int(distance), :, 0] = 0.4*255
    return distance_reading

def draw_distance_sensor_readings(distances):
    color = (1.0, 1.0, 1.0)
    pos_init = np.array([0, 0])
    width = 8
    max_dist = 100
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            vertices = list()
            distance = distances[i, j]
            pos = pos_init + [0+1.5*max_dist*i, 3*width*j]
            # Color
            if distance < max_dist/6:
                color = (0.4, 0.0, 0.0)
            elif distance < max_dist*2/6:
                color = (0.7, 0.0, 0.0)
            elif distance < max_dist*3/6:
                color = (1.0, 0.0, 0.0)
            elif distance < max_dist*4/6:
                color = (0.0, 0.0, 1.0)
            elif distance < max_dist*5/6:
                color = (0.0, 0.0, 0.7)
            else:
                color = (0.0, 0.0, 0.4)
            vertices.append((pos[1] - width, pos[0] + distance, 0))
            vertices.append((pos[1] + width, pos[0] + distance, 0))
            vertices.append((pos[1] + width, pos[0], 0))
            vertices.append((pos[1] - width, pos[0], 0))
            vertices.append((pos[1] - width, pos[0] + distance, 0))
            vertices.append((pos[1] + width, pos[0] + distance, 0))
            vertices.append((pos[1] - width, pos[0], 0))
            vertices.append((pos[1] + width, pos[0], 0))
            draw_box_with_surfaces(vertices, color)


def set_car_viewpoint(car_pos, car_heading, camera_height, car_length):
    car_vel = np.array([np.cos(car_heading), np.sin(car_heading)])
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(car_pos[0] + car_length/2*car_vel[0], car_pos[1]+car_length/2*car_vel[1], 2*camera_height,
              car_pos[0] + (car_length/2+1)*car_vel[0], car_pos[1] + (car_length/2+1)*car_vel[1], 1.75*camera_height, 0, 0, 1)


def set_top_viewpoint(env_width, env_height):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(env_width/2, env_height/2, env_width*0.35, env_width/2, env_height/2, 0, 1, 0, 0)
