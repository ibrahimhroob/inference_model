#!/usr/bin/env python

import os
import sys
import time
import torch
import struct
import argparse
import importlib
import numpy as np

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from loader import Loader

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

THRESHOLD_0 = 0.05
THRESHOLD_1 = 0.85

RAW_POINTCLOUD_TOPIC = '/os_cloud_node/points'
FILTERED_POINTCLOUD_TOPIC = '/cloud_filtered'

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='ktima', help='Trained model')
    return parser.parse_args()


class Stability():
    def __init__(self, model_name):
        rospy.init_node('pointcloud_stability_inference')
        rospy.Subscriber(RAW_POINTCLOUD_TOPIC, PointCloud2, self.callback)

        # Initialize the publisher
        self.pub = rospy.Publisher(FILTERED_POINTCLOUD_TOPIC, PointCloud2, queue_size=10)

        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.device)


        rospy.spin()

    def callback(self, pointcloud_msg):
        pc = ros_numpy.numpify(pointcloud_msg)
        height = pc.shape[0]
        width = pc.shape[1]
        data = np.zeros((height * width, 4), dtype=np.float32)
        data[:, 0] = np.resize(pc['x'], height * width)
        data[:, 1] = np.resize(pc['y'], height * width)
        data[:, 2] = np.resize(pc['z'], height * width)
        data[:, 3] = np.resize(pc['intensity'], height * width)

        # Infere the stability labels
        data = self.infer(data)

        filtered_cloud = self.to_rosmsg(data, pointcloud_msg.header)

        self.pub.publish(filtered_cloud)


    def to_rosmsg(self, data, header):
        filtered_cloud = PointCloud2()
        filtered_cloud.header = header

        # Define the fields for the filtered point cloud
        filtered_fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('intensity', 12, PointField.FLOAT32, 1)]

        filtered_cloud.fields = filtered_fields
        filtered_cloud.is_bigendian = False
        filtered_cloud.point_step = 16
        filtered_cloud.row_step = filtered_cloud.point_step * len(data)
        filtered_cloud.is_bigendian = False
        filtered_cloud.is_dense = True
        filtered_cloud.width = len(data)
        filtered_cloud.height = 1


        # Filter the point cloud based on intensity
        for point in data:
            filtered_cloud.data += struct.pack('ffff', point[0], point[1], point[2], point[3])

        return filtered_cloud


    def load_model(self, device):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, self.model_name)
        sys.path.append(model_path)

        '''HYPER PARAMETER'''
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        MODEL = importlib.import_module('transformer')

        model = MODEL.SPCTReg()
        model.to(device)

        checkpoint = torch.load( os.path.join(model_path, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.eval()   

        rospy.loginfo("Model loaded successfully!")

        return model

    def infer(self, pointcloud):
        
        start_time = time.time()
        FRAME_DATASET = Loader(pointcloud)
        batch_size = FRAME_DATASET.num_windows

        frame_loader = torch.utils.data.DataLoader(FRAME_DATASET, batch_size=batch_size, shuffle=False, num_workers=8,
                                                    pin_memory=True, drop_last=False)
        
        for i, (points, __) in enumerate(frame_loader):
            points = points.float().to(self.device)
            points = points.transpose(2, 1)
            labels = self.model(points)

            points = points.permute(0,2,1).cpu().data.numpy().reshape((-1, 3))
            labels = labels.permute(0,2,1).cpu().data.numpy().reshape((-1, ))

        data = np.column_stack((points, labels))

        data = data[(data[:,3] < THRESHOLD_1) & (data[:,3] >= THRESHOLD_0)]

        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo("Frame inference and filter elapsed time: {:.2f} seconds".format(elapsed_time))

        return data


if __name__ == '__main__':
    args = parse_args()
    stability_node = Stability(args.model)
    





