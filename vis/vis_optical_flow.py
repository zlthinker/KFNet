import numpy as np
import matplotlib.pyplot as plt
import argparse, random, cv2

def read_lines(filepath):
    with open(filepath) as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    return lines

def vis_flow_arrows(image_file1, image_file2, flow_file, thres=100, output_prefix=None):
    image1 = cv2.imread(image_file1)
    image2 = cv2.imread(image_file2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    flow_uv = np.load(flow_file)

    blend_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

    height, width, _ = flow_uv.shape
    print flow_uv.shape
    for h in range(2, height-2, 3):
        for w in range(2, width-2, 3):
            confidence = flow_uv[h, w, 2]
            if confidence < thres:
                continue
            u = flow_uv[h, w, 0]
            v = flow_uv[h, w, 1]
            x = w * 8 + 4
            y = h * 8 + 4
            cv2.line(blend_image, (x, y), (int(x+u*8), int(y+v*8)), (0, 255, 0), 2)

    if output_prefix:
        plt.savefig(output_prefix + '.eps', format='eps')
    else:
        plt.imshow(blend_image)
        plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file1', type=str)
    parser.add_argument('image_file2', type=str)
    parser.add_argument('flow_file', type=str)
    parser.add_argument('--thres', default='100', type=float, help="Confidence threshold")
    parser.add_argument('--output_prefix', type=str)
    args = parser.parse_args()

    vis_flow_arrows(args.image_file1, args.image_file2, args.flow_file, args.output_prefix)

if __name__ == "__main__":
    main()
