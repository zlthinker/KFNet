import numpy as np
import argparse, open3d

def read_lines(filepath):
    with open(filepath) as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_file', type=str, help="Input scene coordinate map in npy format")
    parser.add_argument('--thres', default='20', type=float, help="Confidence threshold")
    args = parser.parse_args()

    map = np.load(args.npy_file)
    coords = map[:, :, 0:3]
    confidences = map[:, :, 3]
    coords = np.reshape(coords, (-1, 3))
    confidences = confidences.flatten().tolist()

    coords_filtered = []
    for i in range(len(confidences)):
        if confidences[i] > args.thres:
            coords_filtered.append(coords[i])
    coords_filtered = np.vstack(coords_filtered)
    print '#points:', coords_filtered.shape[0]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(coords_filtered)
    open3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
