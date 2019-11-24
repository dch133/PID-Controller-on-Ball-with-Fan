import numpy as np
import cv2
import sys
import time

bgr_color = 36, 34, 117
color_threshold = 50  # color range

hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold])
HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])


class PIDController:
    def __init__(self, target_pos):
        self.target_pos = 0.5
        #Values for 2.7g ball
        self.Kp = 3130.25
        self.Ki = 3298.32
        self.Kd = 252.10

        # Values for 10.0g ball
        # self.Kp = 7710.92
        # self.Ki = 5079.66
        # self.Kd = 450.20

        self.bias = 0.0
        self.error_0_to_t = 0.0
        self.prev_error = 0.0
        self.error_at_t = 0.0
        return

    def reset(self):
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0
        self.bias = 0.0
        self.error_0_to_t = 0.0
        self.prev_error = 0.0
        self.error_at_t = 0.0
        return


    def get_fan_rpm(self, image_frame=None):

        _,y,_ = detect_ball(image_frame)
        vertical_ball_position = (485.0-float(y))/485.0 #486 is the height of the tube
        delta_t = 1.0 / 60.0

        #calculate the error function at every frame
        prev_error = self.error_at_t
        self.error_at_t = self.target_pos - vertical_ball_position
        self.error_0_to_t += self.error_at_t*delta_t

        #calculate pid:  u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        _P = self.Kp * self.error_at_t
        _I = self.Ki * self.error_0_to_t
        _D = self.Kd * (self.error_at_t - prev_error) / delta_t
        output = _P + _I + _D

        return output, vertical_ball_position

def detect_ball(frame):
    x, y, radius = -1.0, -1.0, -1.0

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (-1.0, -1.0)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # position of the ball

        # check that the radius is larger than some threshold
        if radius > 10:  # CHANGED
            # outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            # show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[0], center[1], radius
