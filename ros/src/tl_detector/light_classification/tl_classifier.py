
from styx_msgs.msg import TrafficLight
import rospy
import cv2
import numpy as np
import tensorflow as tf
import json
import operator
import os
import timeit

import label_map_util as l_util
import visualization_utils as vis_util


# Traffic Light Classification module. Used other solution
# https://github.com/bguisard/CarND-Capstone-submission/tree/master/ros/src/tl_detector/light_classification

### MODEL CONFIGURATION

MODEL_CONFIG_PATH = './light_classification/model_cfg.json'
MIN_THRESHOLD = 0.5

### END OF MODEL CONFIGURATION

CWD_PATH = os.getcwd()
selected_model = json.load(open(MODEL_CONFIG_PATH))
MODEL_FOLDER = selected_model['model_folder']
MODEL_JSON = selected_model['model_config_file']
MODEL_CFG = os.path.join(CWD_PATH, 'light_classification', MODEL_FOLDER, MODEL_JSON)

# Load model details from file
model_details = json.load(open(MODEL_CFG))

class TLClassifier(object):

    def __init__(self):

        # Gets session, graph, labels and categories

        (self.sess, self.detection_graph, self.label_map,
         self.categories, self.category_index) = self.start_session()

        # TO-DO: Pass blank image to session to initialize the model


    def start_session(self):

        # Load model details from file
        model_details = json.load(open(MODEL_CFG))

        MODEL_NAME = model_details['model_folder']
        MODEL_FILE = model_details['model_filename']
        LABELS_FILE = model_details['labels_filename']
        NUM_CLASSES = model_details['number_of_classes']
        PATH_TO_CKPT = os.path.join(CWD_PATH,'light_classification', 'models', MODEL_NAME, MODEL_FILE)
        PATH_TO_LABELS = os.path.join(CWD_PATH, 'light_classification', 'labels', LABELS_FILE)

        # Load label map
        label_map = l_util.load_labelmap(PATH_TO_LABELS)
        categories = l_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
        category_index = l_util.create_category_index(categories)

        # Starts Session
        # TF Config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph, config=config)

        return sess, detection_graph, label_map, categories, category_index


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Expands image dims as model expects images to have shape: [1, None, None, 3]
        image_np = cv2.resize(image, (300, 300))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Names tensors that will be used 
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Run session
        start_time = timeit.default_timer()
        (boxes, scores, classes) = self.sess.run(
                                     [boxes, scores, classes],
                                     feed_dict={image_tensor:image_np_expanded})
        inference_time = timeit.default_timer() - start_time

        rospy.loginfo("Inference took %s seconds", inference_time)

        # Manipulates output of sess.run
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)

        # Loops through detections and keep the ones above threshold
        results = {}
        for i, box in enumerate(boxes):
            if scores[i] >= MIN_THRESHOLD:
                rospy.loginfo("Class: %s - Score: %s", self.category_index[classes[i]], scores[i])
                if classes[i] not in results:
                    results[classes[i]] = 1
                else:
                    results[classes[i]] += 1

        if results:

            # Selects highes voted class
            highest_voted_idx = max(results.iteritems(), key=operator.itemgetter(1))[0]
            highest_voted = unicode.lower(self.category_index[highest_voted_idx]['name'])
            rospy.loginfo("Predicted light: %s ", highest_voted)
            # Returns styx compatible prediction
            if highest_voted == "green":
                return TrafficLight.GREEN
            elif highest_voted == "yellow":
                return TrafficLight.YELLOW
            elif highest_voted == "red":
                return TrafficLight.RED
            else:
                return TrafficLight.UNKNOWN

        else:
            rospy.loginfo("No lights were detected")
            return TrafficLight.UNKNOWN
