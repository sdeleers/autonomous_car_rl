import numpy as np
import Queue


def perpendicular(v):
    p = [-v[1], v[0]]
    return np.array(p)

def rotate_vector(v, degrees):
    angle_rad = np.deg2rad(degrees)
    r = [v[0]*np.cos(angle_rad) - v[1]*np.sin(angle_rad), v[0]*np.sin(angle_rad) + v[1]*np.cos(angle_rad)]
    return normalize(r)

def normalize(v):
    return v/np.linalg.norm(v)

def angle_between(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    angle = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    cross = np.cross(np.append(v1_u, 0), np.append(v2_u, 0))
    angle *= np.sign(np.dot([0, 0, 1], cross))
    return angle

def get_camera_img2(img, d, N, pos, vel):
    vel = normalize(vel)
    vel_perp = normalize(np.array([-vel[1], vel[0]]))
    p_bottom_left = pos-d/2*vel_perp
    # If indices out of environment, pixel value will be white (a wall around environment)

    camera_img = 255*np.ones((N, N, 3), dtype=np.uint8)
    for i in range(0, N):
        for j in range(0, N):
            raster_indices = np.ceil(p_bottom_left+(N-1-i)*float(d)/(N-1)*vel+j*float(d)/(N-1)*vel_perp).astype(int)
            if all(index >= 0 for index in raster_indices):
                try:
                    camera_img[i, j, :] = img[raster_indices[0], raster_indices[1], :]
                except IndexError:
                    # Indices greater than width/height, these pixels are white (wall).
                    pass
    return camera_img


def car_crashed(car_pos, car_vel, obstacle_pos, car_width, car_height, obst_width, obst_height, env_width, env_height):
    # Check whether crashed against environment wall

    # Check whether crashed against obstacle
    for obst_pos in obstacle_pos:
        if rectangles_overlap(car_pos, car_width, car_height, obst_pos, obst_width, obst_height):
            return True
    return False

# Returns whether two rectangles overlap in 2D.
# Rectangle is specified by its center, width and height.
def rectangles_overlap(c1, w1, h1, c2, w2, h2):
    c1x = c1[0], c1y = c1[1], c2x = c2[0], c2y = c2[1]
    if (c1x + w1 < c2x) or (c2x + w2 < c1x) or (c1y + h1 < c2y) or (Y2 + h2 < c1y):
        return False
    else:
        return True

def get_camera_img(img, d, N, pos, vel):
    vel = normalize(vel)
    vel_perp = normalize(np.array([-vel[1], vel[0]]))
    p_bottom_left = pos - d / 2 * vel_perp
    # If indices out of environment, pixel value will be white (a wall around environment)
    camera_img = 255 * np.ones((N, N), dtype=np.uint8)
    for i in range(0, N):
        for j in range(0, N):
            raster_indices = np.ceil(p_bottom_left + (N - 1 - i) * float(d) / (N - 1) * vel + j * float(d) / (N - 1) * vel_perp).astype(int)
            if all(index >= 0 for index in raster_indices):
                try:
                    camera_img[i, j] = img[raster_indices[0], raster_indices[1]]
                except IndexError:
                    # Indices greater than width/height, these pixels are white (wall).
                    pass
    camera_img = np.repeat(np.expand_dims(camera_img, 2), 3, 2)
    return camera_img


# def get_camera_img(img, d, N, pos, vel):
#     vel = normalize(vel)
#     vel_perp = normalize(np.array([-vel[1], vel[0]]))
#     p_bottom_left = pos - d / 2 * vel_perp
#     # If indices out of environment, pixel value will be white (a wall around environment)
#     camera_img = 255 * np.ones((N, N), dtype=np.uint8)
#     queue = Queue.Queue()
#     workers = [Worker(queue, p_bottom_left, d, pos, vel, vel_perp, img, 0, 16, N), Worker(queue, p_bottom_left, d, pos, vel, vel_perp, img, 16, 32, N), Worker(queue, p_bottom_left, d, pos, vel, vel_perp, img, 32, 48, N), Worker(queue, p_bottom_left, d, pos, vel, vel_perp, img, 48, 64, N)]
#     for worker in workers:
#         worker.start()
#
#     for worker in workers:
#         worker.join()
#
#     # while queue.qsize():
#     #     print queue.get()
#
#     for i in range(4):
#         camera_img[16*i:16*(i+1), :] = queue.get()[0]
#
#     camera_img = np.repeat(np.expand_dims(camera_img, 2), 3, 2)
#     return camera_img
#
# class Worker(threading.Thread):
#     def __init__(self, queue, p_bottom_left, d, pos, vel, vel_perp, img, start_index, stop_index, N):
#         super(Worker, self).__init__()
#         self._queue = queue
#         self.N = N
#         self.start_index = start_index
#         self.stop_index = stop_index
#         self.img = img
#         self.p_bottom_left = p_bottom_left
#         self.d = d
#         self.pos = pos
#         self.vel = vel
#         self.vel_perp = vel_perp
#
#     def run(self):
#         N = self.N
#         d = self.d
#         pos = self.pos
#         vel = self.vel
#         vel_perp = self.vel_perp
#         part_img = 255 * np.ones((16, 64), dtype=np.uint8)
#         for i in range(self.start_index, self.stop_index):
#             for j in range(0, 64):
#                 raster_indices = np.ceil(self.p_bottom_left + (N - 1 - i) * float(d) / (N - 1) * vel + j * float(d) / (N - 1) * vel_perp).astype(int)
#                 if all(index >= 0 for index in raster_indices):
#                     try:
#                         part_img[i, j] = self.img[raster_indices[0], raster_indices[1]]
#                     except IndexError:
#                         # Indices greater than width/height, these pixels are white (wall).
#                         pass
#         self._queue.put(part_img)

# Only works for axis aligned rectangle obstacles, and circular car
def circular_car_crashed(car_pos, radius, obst_centers, obst_width, obst_height):
    for obst_center in obst_centers:
        circle_distance_x = abs(car_pos[0] - obst_center[0]);
        circle_distance_y = abs(car_pos[1] - obst_center[1]);
        if circle_distance_x > obst_width/2+radius:
            continue
        elif circle_distance_y > obst_height/2+radius:
            continue
        elif circle_distance_x <= obst_width/2:
            return True
        elif circle_distance_y <= obst_height/2:
            return True
        corner_distance_sq = (circle_distance_x - obst_width/2)**2 + (circle_distance_y - obst_height/2)**2
        if corner_distance_sq <= radius**2:
            return True
    return False

def distance_nearest_object_in_all_directions(p0, vel, radius, env_width, env_height, screen):
    distances = []
    for i in range(-90, 90):
        direction = rotate_vector(vel, i)
        distance = distance_nearest_object(p0+1.3*radius*direction, direction, env_width, env_height, screen)
        distances.append(distance)
    return np.array(distances)

def circular_car_outside_environment(p, radius, env_width, env_height):
    p_x = int(round(p[0]))
    p_y = int(round(p[1]))
    if (p_x >= radius) and (p_x < env_width-radius) and (p_y >= radius) and (p_y < env_height-radius):
        return False
    else:
        return True

def point_outside_environment(p, env_width, env_height):
    p_x = int(round(p[0]))
    p_y = int(round(p[1]))
    if (p_x > 0) and (p_x < env_width) and (p_y > 0) and (p_y < env_height):
        return False
    else:
        return True


def point_inside_obstacle(p, screen):
    try:
        color = screen.get_at((int(round(p[0])), int(round(p[1]))))
    except IndexError:
        return False
    return color[0] == 255


def distance_nearest_object(p0, direction, env_width, env_height, screen):
    distance = 0
    p = p0
    for i in range(0, 1000):
        distance = i
        if point_outside_environment(p, env_width, env_height):
            break
        elif point_inside_obstacle(p, screen):
            break
        p += direction
    return distance
