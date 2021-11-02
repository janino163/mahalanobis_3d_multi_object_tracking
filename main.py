# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from scipy.optimize import linear_sum_assignment as linear_assignment
from filterpy.kalman import KalmanFilter
from .utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from .covariance import Covariance
import json
# from nuscenes import NuScenes
# from nuscenes.eval.common.data_classes import EvalBoxes
# from nuscenes.eval.tracking.data_classes import TrackingBox 
# from nuscenes.eval.detection.data_classes import DetectionBox 
from pyquaternion import Quaternion
from tqdm import tqdm
import torch 

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox3D, info, covariance_id=0, track_score=None, tracking_name='car', use_angular_velocity=False):
        """
        Initialises a tracker using initial bounding box.
        """
        """              
        observation: 
          [x, y, z, dx, dy, dz, heading]
        state:
          [x, y, z, rot_y, dx, dy, dz, x_dot, y_dot, z_dot]
        """
        #define constant velocity model
        if not use_angular_velocity:
            self.kf = KalmanFilter(dim_x=10, dim_z=7)       
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                                [0,1,0,0,0,0,0,0,1,0],
                                [0,0,1,0,0,0,0,0,0,1],
                                [0,0,0,1,0,0,0,0,0,0],  
                                [0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,0,0,1]])     
    
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                                [0,1,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0]])
        else:
            # with angular velocity
            self.kf = KalmanFilter(dim_x=11, dim_z=7)       
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
                                [0,1,0,0,0,0,0,0,1,0,0],
                                [0,0,1,0,0,0,0,0,0,1,0],
                                [0,0,0,1,0,0,0,0,0,0,1],  
                                [0,0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,0,0,0,1]])     
     
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
                                [0,1,0,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0,0]])

        # Initialize the covariance matrix, see covariance.py for more details
        if covariance_id == 0: # exactly the same as AB3DMOT baseline
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
            self.kf.P[7:,7:] *= 1000. #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
            self.kf.P *= 10.
    
            # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
            self.kf.Q[7:,7:] *= 0.01
        elif covariance_id == 1: # for kitti car, not supported
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P
            self.kf.Q = covariance.Q
            self.kf.R = covariance.R
        elif covariance_id == 2: # for nuscenes
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P[tracking_name]
            self.kf.Q = covariance.Q[tracking_name]
            self.kf.R = covariance.R[tracking_name]
            if not use_angular_velocity:
                self.kf.P = self.kf.P[:-1,:-1]
                self.kf.Q = self.kf.Q[:-1,:-1]
            else:
                assert(False)

        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1           # number of total hits including the first detection
        self.hit_streak = 1     # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info        # other info
        self.track_score = track_score
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity

    def update(self, bbox3D, info): 
        """ 
        Updates the state vector with observed bbox.
        bbox3D: [x, y, z, dx, dy, dz, heading]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1          # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[-1]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[-1] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
      
        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2
    
        ######################### 

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info

    def predict(self):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()      
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7, ))

def angle_in_range(angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle

def diff_orientation_correction(det, trk):
    '''
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    '''
    diff = det - trk
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    diff = angle_in_range(diff)
    return diff

def greedy_match(distance_matrix):
    '''
    Find the one-to-one matching using greedy allgorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    '''
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices
 

def associate_detections_to_trackers(detections,trackers,iou_threshold=0.1, 
  use_mahalanobis=False, dets=None, trks=None, trks_S=None, mahalanobis_threshold=0.1, print_debug=False, match_algorithm='greedy'):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    detections:  N x 8 x 3
    trackers:    M x 8 x 3

    dets: N x 7
    trks: M x 7
    trks_S: N x 7 x 7

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    distance_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    if use_mahalanobis:
        assert(dets is not None)
        assert(trks is not None)
        assert(trks_S is not None)

    if use_mahalanobis and print_debug:
        print('dets.shape: ', dets.shape)
        print('dets: ', dets)
        print('trks.shape: ', trks.shape)
        print('trks: ', trks)
        print('trks_S.shape: ', trks_S.shape)
        print('trks_S: ', trks_S)
        S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
        S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]# 7
        print('S_inv_diag: ', S_inv_diag)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            if use_mahalanobis:
                S_inv = np.linalg.inv(trks_S[t]) # 7 x 7
                diff = np.expand_dims(dets[d] - trks[t], axis=1) # 7 x 1
                # manual reversed angle by 180 when diff > 90 or < -90 degree
                corrected_angle_diff = diff_orientation_correction(dets[d][-1], trks[t][3])
                diff[-1] = corrected_angle_diff
                distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
            else:
                iou_matrix[d,t] = boxes_bev_iou_cpu(dets[d],trks[t])[0]             # det: 1 x 7, trk: 1 x 7
                distance_matrix = -iou_matrix

    if match_algorithm == 'greedy':
        matched_indices = greedy_match(distance_matrix)
    elif match_algorithm == 'pre_threshold':
        if use_mahalanobis:
            to_max_mask = distance_matrix > mahalanobis_threshold
            distance_matrix[to_max_mask] = mahalanobis_threshold + 1
        else:
            to_max_mask = iou_matrix < iou_threshold
            distance_matrix[to_max_mask] = 0
            iou_matrix[to_max_mask] = 0
        matched_indices = linear_assignment(distance_matrix)      # houngarian algorithm
    else:
        matched_indices = linear_assignment(distance_matrix)      # houngarian algorithm

    if print_debug:
        print('distance_matrix.shape: ', distance_matrix.shape)
        print('distance_matrix: ', distance_matrix)
        print('matched_indices: ', matched_indices)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        match = True
    if use_mahalanobis:
        if distance_matrix[m[0],m[1]] > mahalanobis_threshold:
            match = False
    else:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            match = False
    if not match:
        unmatched_detections.append(m[0])
        unmatched_trackers.append(m[1])
    else:
        matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    if print_debug:
        print('matches: ', matches)
        print('unmatched_detections: ', unmatched_detections)
        print('unmatched_trackers: ', unmatched_trackers)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class AB3DMOT(object):
    def __init__(self,covariance_id=0, max_age=2,min_hits=3, tracking_name='car', use_angular_velocity=False, tracking_openpcdet=False):
        """              
        observation: 
          [x, y, z, dx, dy, dz, heading]
        state:
          [x, y, z, rot_y, dx, dy, dz, x_dot, y_dot, z_dot]
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_openpcdet = tracking_openpcdet

    def update(self,dets_all, match_distance, match_threshold, match_algorithm):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x, y, z, dx, dy, dz, heading],[x, y, z, dx, dy, dz, heading],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
        print_debug = False

        self.frame_count += 1

        trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
        for t in reversed(to_del):
            self.trackers.pop(t)


        dets_8corner = [convert_3dbox_to_8corner(det_tmp, match_distance == 'iou' and self.tracking_openpcdet) for det_tmp in dets]
        if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
        else: dets_8corner = []

        trks_8corner = [convert_3dbox_to_8corner(trk_tmp, match_distance == 'iou' and self.tracking_openpcdet) for trk_tmp in trks]
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]

        if len(trks_8corner) > 0: 
            trks_8corner = np.stack(trks_8corner, axis=0)
            trks_S = np.stack(trks_S, axis=0)
        if match_distance == 'iou':
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, iou_threshold=match_threshold, print_debug=print_debug, match_algorithm=match_algorithm)
        else:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, use_mahalanobis=True, dets=dets, trks=trks, trks_S=trks_S, mahalanobis_threshold=match_threshold, print_debug=print_debug, match_algorithm=match_algorithm)
   
        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
                trk.update(dets[d,:][0], info[d, :][0])
                detection_score = info[d, :][0][-1]
                trk.track_score = detection_score

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:        # a scalar of index
            detection_score = info[i][-1]
            track_score = detection_score
            trk = KalmanBoxTracker(dets[i,:], info[i, :], self.covariance_id, track_score, self.tracking_name, use_angular_velocity) 
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()      # bbox location
            d = d[self.reorder_back]

            if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
                ret.append(np.concatenate((d, [trk.id+1], trk.info[:-1], [trk.track_score])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)      # x, y, z, theta, l, w, h, ID, other info, confidence
        return np.empty((0,15 + 7))

NUSCENES_TRACKING_NAMES = [
  'bicycle',
  'bus',
  'car',
  'motorcycle',
  'pedestrian',
  'trailer',
  'truck'
]

def track_Ithaca_365(scene, save_path, covariance_id=2, match_distance="iou", match_threshold=11, match_algorithm="greedy", save_root=0, use_angular_velocity=1):
    results = {}
    total_time = 0.0
    total_frames = 0
    mot_trackers = {tracking_name: AB3DMOT(covariance_id, tracking_name=tracking_name, use_angular_velocity=use_angular_velocity, tracking_openpcdet=True) for tracking_name in NUSCENES_TRACKING_NAMES}
    annotation_frames = scene["boxes"]
    for index_, frame in enumerate(annotation_frames):
        dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
        info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
        for i, box in enumerate(frame):
            if box.label not in ['pedestrian', 'bus', 'car']:
                continue
            dets[box.label].append(box.get_detection())
            info[box.label].append(box.get_detection())
            
        dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])}
        for tracking_name in NUSCENES_TRACKING_NAMES}
        
        total_frames += 1
        start_time = time.time()
        for tracking_name in NUSCENES_TRACKING_NAMES:
#             if dets_all[tracking_name]['dets'].shape[0] > 0:
            trackers = mot_trackers[tracking_name].update(dets_all[tracking_name], match_distance, match_threshold, match_algorithm)
        temp_positions = []
        for i in range(trackers.shape[0]):
            trackers[i]
        
        
           
    

if __name__ == '__main__':
    pass
