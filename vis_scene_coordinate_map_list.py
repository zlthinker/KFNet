import argparse, open3d, math, time
import numpy as np
from vis_scene_coordinate_map import read_npy_as_pcd

def read_lines(filepath):
    with open(filepath) as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    return lines

def rotation_matrix(angle, direction, point=None):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    print direction, direction.dtype
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_list', type=str, help="Input ply list")
    parser.add_argument('--step', default='1', type=int, help="The step of frame sampling")
    parser.add_argument('--thres', default='20', type=float, help="Confidence threshold")
    parser.add_argument('--rotate', default='0', type=float, help="Angle to rotate the point clouds")
    args = parser.parse_args()

    npy_files = read_lines(args.npy_list)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    rotation = rotation_matrix(math.pi * args.rotate / 180.0, np.array([1.0, 0, 0]))
    print 'rotation', rotation

    line_set = open3d.geometry.LineSet()
    line_set.transform(rotation)

    for i in range(0, len(npy_files), args.step):
        npy_file = npy_files[i]
        print npy_file
        pcd = read_npy_as_pcd(npy_file, args.thres)
        pcd.transform(rotation)

        vis.add_geometry(pcd)
        vis.update_geometry(None)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05 * args.step)
    vis.destroy_window()

if __name__ == "__main__":
    main()