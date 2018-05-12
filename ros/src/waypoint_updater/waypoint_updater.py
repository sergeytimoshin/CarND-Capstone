#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 70 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        
        rospy.init_node('waypoint_updater')

        self.slow_down = False

        self.base_lane = None
        self.base_waypoints = None
        self.pose = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.vehicle_velocity = 0
        self.previous_velocity = 0

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        # self.final_waypoints_pub.publish(lane)


        self.loop()
        # rospy.spin()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.vehicle_velocity and self.waypoint_tree:
                self.publish_waypoints()

            self.previous_velocity = self.vehicle_velocity
            rate.sleep()

    def get_closest_waypoint_idx(self):        
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_lane.header
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx != -1 and closest_idx <= self.stopline_wp_idx <= farthest_idx:
            self.slow_down = True
            waypoints = self.decelerate(waypoints, closest_idx)
        elif self.slow_down:
            self.slow_down = False

        if not self.slow_down:
            if abs(self.vehicle_velocity - self.get_waypoint_velocity(waypoints[0])) > 1.0:
                start_velocity = max(self.previous_velocity + 0.5, self.vehicle_velocity)
                waypoints = self.accelerate(waypoints, start_velocity)

        lane.waypoints = waypoints
        return lane

    def accelerate(self, waypoints, start_velocity):
        lane_waypoints = []
        prev_velocity = start_velocity
        for i, waypoint in enumerate(waypoints):
            p = Waypoint()
            p.pose = waypoint.pose

            velocity = prev_velocity + 0.5
            prev_velocity = velocity

            if velocity < 0.5:
                velocity = 0.5

            velocity = min(velocity, self.get_waypoint_velocity(waypoint))
            p.twist.twist.linear.x = velocity
            lane_waypoints.append(p)
        return lane_waypoints


    def decelerate(self, waypoints, closest_idx):
        tl_idx = self.stopline_wp_idx - closest_idx - 2
        # rospy.logwarn("waypoints.count = {0} ## tl_idx = {1}".format(len(waypoints), tl_idx))

        velocity = self.vehicle_velocity
        deccel = velocity / (self.distance(waypoints, 0, tl_idx) + 1.1)

        rospy.logwarn("vel = {0} ## deccel = {1}".format(velocity, deccel))

        processed_waypoints = []
        for i, waypoint in enumerate(waypoints):
            p = Waypoint()
            p.pose = waypoint.pose

            if i >= tl_idx:
                velocity = 0

            distance = self.distance(waypoints, i, tl_idx)
            if distance < 40:
                velocity = distance * deccel * 0.1

            velocity = min(velocity, self.get_waypoint_velocity(waypoint))

            if velocity <= 3:
               velocity = 0

            rospy.logwarn("new velocity = {0}".format(velocity))
            p.twist.twist.linear.x = velocity
            processed_waypoints.append(p)
        return processed_waypoints

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_lane = waypoints
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def velocity_cb(self, velocity):
        self.vehicle_velocity = velocity.twist.linear.x

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        n = len(waypoints)
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2):
            dist += dl(waypoints[i % n].pose.pose.position, waypoints[(i + 1) % n].pose.pose.position)
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
