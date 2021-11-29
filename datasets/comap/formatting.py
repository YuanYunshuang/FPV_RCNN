import shutil, os, sys
import numpy as np
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.spatial import Delaunay
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
import open3d as o3d
from datasets.comap import LABEL_COLORS


from vlib.visulization import draw_points_boxes_plt_2d
from utils.tfs import rotate_points_along_z


def gt_bbox_for_all_cavs(data_path):
    bbox_out = os.path.join(data_path, "bbox_global_z")
    files = glob(os.path.join(data_path, 'cloud_ego' , '*.bin'))
    for file in tqdm(files):
        name = file.rsplit('/')[-1].rsplit('.')[0]
        gt_dir = os.path.join(bbox_out, name)
        os.makedirs(gt_dir, exist_ok=True)
        info_file = file.replace('cloud_ego', 'poses_global').replace('bin', 'txt')

        vehicles_info = np.loadtxt(info_file, dtype=str)[:, [0, 2, 3, 4, 8, 9, 10, 7]].astype(np.float)
        for v_info in vehicles_info:
            cur_id = int(v_info[0])
            gt_boxes = np.copy(vehicles_info[:, 1:])
            gt_boxes[:, :2] = gt_boxes[:, :2] - v_info[None, 1:3]
            # gt_boxes[:, 2] = gt_boxes[:, 2] - v_info[-2] / 2 - 0.3
            # output gt boxes files
            np.savetxt(os.path.join(gt_dir, '{:06d}.txt'.format(cur_id)), gt_boxes, fmt="%.3f")
            ##############for visulization only#################
            # if cur_id==0:
            #     cloud = np.fromfile(file, 'float32').reshape(-1, 3)
            # else:
            #     coop_file = os.path.join(file.replace('cloud_ego', 'cloud_coop')[:-4], '{:06d}.bin'.format(cur_id))
            #     cloud = np.fromfile(coop_file, 'float32').reshape(-1, 3)
            # cloud[:, :2] *= -1
            # cloud_rot = rotate_points_along_z(cloud, v_info[-1])
            # draw_points_boxes_plt_2d(pc_range, cloud_rot, boxes_gt=gt_boxes)
            ##############for visulization only#################


def get_global_boxes(info_dir, vtypes_file, out_path, ego_id_file):
    ego_ids = np.load(ego_id_file, allow_pickle=True).item()
    vtypes_cls, vtypes_size = read_vtypes(vtypes_file)
    for info_file in glob(info_dir + '/*.csv'):
        cur_junc = info_file.rsplit('/')[-1][:-4]
        with open(info_file, 'r') as fh:
            infos = fh.readlines()
            infos.pop(0)
        # read vehicle info to dict
        info_dict = {}
        for s in infos:
            info = s.strip().split(',')
            if info[0] not in info_dict:
                info_dict[info[0]] = {}
            info_dict[info[0]][info[2]] = info[3:] + [info[1]]
        # change metric and tranform to ego-lidar CS
        for frame, data in tqdm(info_dict.items()):
            ss = '{} ' * 2 + '{:.3f} ' * 9 + '\n'
            with open(os.path.join(out_path, cur_junc[1:] + '_' + frame + '.txt'), 'w') as flbl:
                for k, values in data.items():
                    # values: x,y,z,rx,ry,rz,l,w,h,type_id
                    type_id = values[-1]
                    vid = '000000' if k==ego_ids[cur_junc] else k
                    v_class = vtypes_cls[type_id]
                    v_size = [float(s) for s in values[6:9]]
                    tf_g = np.array(values[:6]).astype(np.float)
                    tf_g[3:] = - tf_g[3:] / 180 * np.pi
                    tf_g[2] = tf_g[2] + v_size[2] / 2
                    tf_g[1] = - tf_g[1]
                    flbl.write(ss.format(vid, v_class, *tf_g, *v_size))


def read_vtypes(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    vtypes_cls = {}
    vtypes_size = {}
    for vt in root.findall('vType'):
        vtypes_cls[vt.attrib['id']] = vt.attrib['vClass']
        vtypes_size[vt.attrib['id']] = [vt.attrib['length'],
                                        vt.attrib['width'],
                                        vt.attrib['height']]
    return vtypes_cls, vtypes_size


def copy_csvs(data_path):
    ego_ids = {}
    for file in glob("/media/hdd/ophelia/koko/data/simulation_20veh_60m/*/info.csv"):
        junc = file.split('/')[-2]
        shutil.copy(file, os.path.join(data_path, 'csvs', junc + '.csv'))
        ego_id = glob(file.rsplit('/', 1)[0] + '/*.ego')[0].rsplit('/')[-1][:-4]
        ego_ids[junc] = ego_id
    np.save(os.path.join(data_path, 'csvs', 'ego_ids.npy'), ego_ids, allow_pickle=True)


def read_net(net_filename):
    """
    Read sumo net.xml file and find points of roads lanes.
    @param net_filename: file path to the sumo net file
    @return:
        lines: list of [x, y]
    """
    netoffset = [270.80, 200.32]
    tree = ET.parse(net_filename)
    root = tree.getroot()
    edges = root.findall("edge")
    points = []
    for edge in edges:
        for lane in edge.findall("lane"):
            # remove sidewalk and shoulder lanes
            allow = lane.attrib["allow"] if 'allow' in lane.attrib else None
            disallow = lane.attrib["disallow"] if 'disallow' in lane.attrib else None
            if 'pedestrian'==allow or 'all'==disallow:
                continue
            coordinate = lane.attrib["shape"].split(' ')
            p = coordinate[0].split(',')
            last_point = [float(p[0]) - netoffset[0], float(p[1]) - netoffset[1]]
            # points.append(last_point)
            for i in range(1, len(coordinate)):
                p = coordinate[i].split(',')
                cur_point = [float(p[0]) - netoffset[0], float(p[1]) - netoffset[1]]
                dist = np.sqrt((cur_point[0] - last_point[0])**2 + (cur_point[1] - last_point[1])**2)
                if dist < 3:
                    continue
                elif dist > 6:
                    n_split = int(dist // 6 + 1)
                    delta_x = (cur_point[0] - last_point[0]) / n_split
                    delta_y = (cur_point[1] - last_point[1]) / n_split
                    for n in range(1, n_split):
                        interp_point = [last_point[0] + n * delta_x, last_point[1] + n * delta_y]
                        points.append(interp_point)
                points.append(cur_point)
                last_point = cur_point
            # lane_type = lane.attrib["type"] if "type" in edge.attrib else "internal"
    return np.array(points)


def delaunay_graph(points, plot=False):
    tri = Delaunay(points)
    n = len(points)
    graph = np.zeros((n, n))

    # plt.plot(points[:, 0], points[:, 1], 'k.')
    for i in range(tri.nsimplex):
        vertices = tri.simplices[i]
        edges = list(combinations(vertices, 2))
        for e in edges:
            dist = np.linalg.norm(points[e[0]] - points[e[1]])
            if dist < 8:
                graph[e[0], e[1]] = dist
                graph[e[1], e[0]] = dist
                if plot:
                    plt.plot(points[e, 0], points[e, 1], 'y')
    return graph


def find_shortest_path(graph, start_point_idx, end_point_inds):
    """
    Given one start point and a list of end points, return the distances of the start point to the end points
    """
    graph = csr_matrix(graph)
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False,
                                              indices=start_point_idx, return_predecessors=True)
    distances = dist_matrix[end_point_inds]
    return distances


def filter_gt_boxes_via_driving_dist(points, boxes, center_veh_loc, frame, obs_r=57.6, plot=False):
    points_local = points - center_veh_loc[None, :]
    points_extended = np.concatenate([points_local, boxes[:, :2]], axis=0)
    dists = np.linalg.norm(points_extended, axis=1)
    boxes = boxes[np.linalg.norm(boxes[:, :2], axis=1) <= obs_r]
    mask = dists <= obs_r
    masked_dist = dists[mask]
    masked_points = points_extended[mask]
    start_point_idx = np.argmin(masked_dist)
    end_points_dists = np.linalg.norm(masked_points[None, :, :] - boxes[:, :2][:, None, :], axis=2)
    end_points_inds = np.argmin(end_points_dists, axis=1)
    graph = delaunay_graph(masked_points, plot)
    if plot:
        plt.plot(boxes[:, 0], boxes[:, 1], 'bo')
        plt.plot([0], [0], 'r*', markersize=15)
        plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
    graph_dists = find_shortest_path(graph, start_point_idx, end_points_inds)
    mask_not_inf = np.logical_not(np.isinf(graph_dists))
    mask_in_r = graph_dists[mask_not_inf] <= obs_r
    inds = np.where(mask_not_inf)[0][mask_in_r]
    boxes_selected = boxes[inds]
    if plot:
        plt.plot(boxes_selected[:, 0], boxes_selected[:, 1], 'go')
        plt.savefig("/media/hdd/ophelia/tmp/gt_boxes_1/{:s}.png".format(frame))
        plt.close()
    return boxes_selected, inds


def filter_gt_boxes(data_path):
    net_file = "/media/hdd/ophelia/fusion_ssd/datasets/Town05.net.xml"
    out_path = os.path.join(data_path, 'bbox_all_Dr') # in drivable range
    os.makedirs(out_path, exist_ok=True)
    points = read_net(net_file)
    for frame_dir in tqdm(os.listdir(os.path.join(data_path, 'bbox_all'))):
        os.makedirs(os.path.join(out_path, frame_dir), exist_ok=True)
        info_file = os.path.join(os.path.join(data_path, 'poses_global'), frame_dir + '.txt')
        vehicles_info = np.loadtxt(info_file, dtype=str)[:, [0, 2, 3]].astype(np.float)
        vehicles_info_dict = {int(v_info[0]): v_info[1:] for v_info in vehicles_info}
        for bbox_file in glob(os.path.join(data_path, 'bbox_all', frame_dir + '/*.txt')):
            v_id = bbox_file.split('/')[-1][:-4]
            bbox = np.loadtxt(bbox_file, dtype=np.float)
            center_veh_loc = vehicles_info_dict[int(v_id)]
            selected_boxes = filter_gt_boxes_via_driving_dist(points, bbox, center_veh_loc, frame=frame_dir)
            np.savetxt(os.path.join(out_path, frame_dir, bbox_file.split('/')[-1]), selected_boxes,
                       fmt="%.3f")


def check_gt_bboxes(datapath):
    obs_r = 57.6
    pc_range = np.array([-obs_r, -obs_r, -3, obs_r, obs_r, 1])
    def _mask_points_in_range(points):
        return points[np.linalg.norm(points[:, :2], axis=1) <= obs_r]
    for file in glob(os.path.join(datapath, 'cloud_ego', '*.bin')):
        # check ego gt
        cloud_ego = np.fromfile(file, dtype="float32").reshape(-1, 4)
        cloud_ego = _mask_points_in_range(cloud_ego)
        bbox_file = file.replace('cloud_ego', 'bbox_all')[:-4] + '/000000.txt'
        bbox_Dr_file = file.replace('cloud_ego', 'bbox_all_Dr')[:-4] + '/000000.txt'
        gt_boxes = np.loadtxt(bbox_file, dtype=np.float)
        gt_boxes_Dr = np.loadtxt(bbox_Dr_file, dtype=np.float)
        vehicle_info_file = file.replace('cloud_ego', 'poses_global').replace('bin', 'txt')
        vehicles_info = np.loadtxt(vehicle_info_file, dtype=str)[:, [0, 2, 3, 4, 8, 9, 10, 7]].astype(np.float)
        vehicles_info_dict = {int(v_info[0]): v_info[1:] for v_info in vehicles_info}
        cloud_ego = rotate_points_along_z(cloud_ego, vehicles_info_dict[0][-1])
        #gt_boxes[:, [0, 1]] = - gt_boxes[:, [0, 1]]
        draw_points_boxes_plt_2d(pc_range, points=cloud_ego, boxes_gt=gt_boxes, boxes_pred=gt_boxes_Dr)
        print('waite')


def read_coop_clouds(datapath):
    raw_data_path = '/media/hdd/ophelia/koko/data/simulation_20veh_60m'
    dirs = os.listdir(os.path.join(datapath, 'cloud_coop_no_label'))
    for d in tqdm(dirs):
        files = glob(os.path.join(datapath, 'cloud_coop_no_label', d, '*.bin'))
        os.makedirs(os.path.join(datapath, 'cloud_coop', d))
        for f in files:
            # pcd_no_label = np.fromfile(f, dtype="float32").reshape(-1, 3)
            fl = f.split('/')
            junc, frame = tuple(fl[-2].split('_'))
            v_id = fl[-1][:-4]
            pcd_file = os.path.join(raw_data_path, 'j' + junc, v_id, 'lidar_sem', frame + '.pcd')
            pcd = o3d.io.read_point_cloud(pcd_file)
            # points_np = np.array(pcd.points).astype(np.float32)
            save_cloud_to_bin(pcd, f.replace('cloud_coop_no_label', 'cloud_coop'))


def save_cloud_to_bin(cloud, filename):
    ## get points' coordinates
    points_np = np.array(cloud.points).astype(np.float32)
    ## get point-wise semantic label
    colors_np = (np.array(cloud.colors) * 255).astype(np.uint8)
    dists = np.abs(colors_np[:, :, None] - np.array(list(LABEL_COLORS.values())).T[None, :, :]).sum(axis=1)
    labels = np.argmin(dists, axis=1).reshape(-1, 1)
    ## append label to the point as the 4-th element and add the labeled points to list
    points = np.concatenate([points_np, labels.astype(np.float32)], axis=1)
    ## write binary file
    points.astype('float32').tofile(filename)


if __name__=="__main__":
    data_path = "/media/hdd/ophelia/koko/data/synthdata_20veh_60m"
    # vtypes_file = "../assets/data_carlavtypes.rou.xml"
    # get_global_boxes(
    #     os.path.join(data_path, 'csvs'),
    #     vtypes_file,
    #     os.path.join(data_path, "poses_global"),
    #     os.path.join(data_path, "csvs/ego_ids.npy")
    # )
    # gt_bbox_for_all_cavs(data_path)
    # check_gt_bboxes(data_path)
    # filter_gt_boxes(data_path)
    read_coop_clouds(data_path)

