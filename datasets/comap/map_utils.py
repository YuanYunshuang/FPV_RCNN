import os.path

from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from skimage.io import imsave, imread


def read_net(net_filename):
    """
    Read sumo net.xml file and extracts the geometric lines of roads lanes.
    @param net_filename: file path to the sumo net file
    @return:
        lines: Numpy array of lines, each raw indicates [x1, y1, x2, y2, w],
        where x1, y1 is the start point, x2, y2 is the
        end point of the line, w is the width of the lane
    """
    tree = ET.parse(net_filename)
    root = tree.getroot()
    edges = root.findall("edge")
    lines = []
    for edge in edges:
        for lane in edge.findall("lane"):
            # remove sidewalk and shoulder lanes
            allow = lane.attrib["allow"] if 'allow' in lane.attrib else None
            disallow = lane.attrib["disallow"] if 'disallow' in lane.attrib else None
            if 'pedestrian'==allow or 'all'==disallow:
                continue
            width = lane.attrib["width"]
            coordinate = lane.attrib["shape"].split(' ')
            for i in range(0, len(coordinate) - 1):
                p1 = coordinate[i].split(',')
                p2 = coordinate[i+1].split(',')
                lines.append([float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), float(width)])
            # lane_type = lane.attrib["type"] if "type" in edge.attrib else "internal"
    return np.array(lines)


def get_map(vec_map_path):
    """
    Transform vector map/net to raster binary map
    @param map_path: file path to the sumo net file
    @return: transformed binary map, 1 in the map indicates drivable areas/roads
    """
    netoffset = [270.80,200.32]
    origBoundary= [-400,-320, 320, 320] # [-270.80,-200.32,202.31,199.12]
    lines = read_net(vec_map_path)
    x1 = lines[:, 0] - netoffset[0]
    y1 = lines[:, 1] - netoffset[1]
    x2 = lines[:, 2] - netoffset[0]
    y2 = lines[:, 3] - netoffset[1]
    w = lines[:, 4] + 0.2
    thetas = np.arctan2(y2-y1, x2-x1)
    #index = np.where(np.abs(thetas-0.8)< 0.2)[0]
    mean = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], axis=1)
    l = np.sqrt(np.square(x2-x1) + np.square(y2-y1))
    # map_size = (np.array([origBoundary[2] - origBoundary[0],  origBoundary[3] -origBoundary[1]]) / 0.1).astype(np.int)
    # Map = np.zeros(map_size)
    map_size = (np.array([origBoundary[2] - origBoundary[0], origBoundary[3] - origBoundary[1]]) / 0.1).astype(np.int)
    Map = np.zeros(map_size)
    for i in tqdm(range(len(lines))):
        x = np.arange(-l[i]/2 + 0.05, l[i]/2, 0.1)
        y = np.arange(-w[i]/2 + 0.05, w[i]/2, 0.1)
        xx, yy = np.meshgrid(x, y)
        points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        theta = thetas[i]
        Mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points = np.dot(points, Mat) + mean[i]

        inds = np.floor((points - np.array(origBoundary[:2])) / 0.1).astype(np.int) #
        inds_x = np.clip(inds[:, 0], a_min=0, a_max=map_size[0] - 1)
        inds_y = np.clip(inds[:, 1], a_min=0, a_max=map_size[1] - 1)

        Map[inds_x, inds_y] = 1
    # smothing the map
    Map = Map.astype(np.uint8) * 255
    imsave(os.path.join(vec_map_path.rsplit('/')[0], "binary_map.png"), Map)
    return Map


if __name__=="__main__":
    if os.path.exists("../assets/binary_map.png"):
        img = imread('assets/binary_map.png')
    else:
        img = get_map("../Town05.net.xml")

    # smothing the map
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(80, 80))
    closed_map = cv2.morphologyEx(img.astype(np.float) / 255, cv2.MORPH_CLOSE, kernel)
    map_out = closed_map.astype(np.uint8) * 255
    imsave('/media/hdd/ophelia/fusion_ssd/datasets/binary_map_closed.png', map_out)