from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):

    def get_path_to_graph(self, is_simulator):
        if is_simulator:
            return 'light_classification/models/sim_frozen_inference_graph.pb'
        else:
            return 'light_classification/models/udacity_frozen_inference_graph.pb'

    def __init__(self, is_simulator):

        self.graph = tf.Graph()
        self.threshold = .5

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.get_path_to_graph(is_simulator), 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if scores[0] > self.threshold:
            if classes[0] == 1:
                rospy.logwarn('GREEN')
                return TrafficLight.GREEN
            elif classes[0] == 2:
                rospy.logwarn('RED')
                return TrafficLight.RED
            elif classes[0] == 3:
                rospy.logwarn('YELLOW')
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
