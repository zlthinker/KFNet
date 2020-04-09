import numpy as np
import matplotlib.pyplot as plt
import argparse, random, cv2, os

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

    blend_image = cv2.addWeighted(image1, 0.7, image2, 0.3, 0)

    height, width, _ = flow_uv.shape
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
        blend_image = cv2.cvtColor(blend_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_prefix + '.png', blend_image)
    else:
        plt.imshow(blend_image)
        plt.show()
        plt.close('all')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('flow_file_list', type=str)
    parser.add_argument('image_list', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('--thres', default='100', type=float, help="Confidence threshold")
    args = parser.parse_args()

    flow_files = read_lines(args.flow_file_list)
    image_files = read_lines(args.image_list)
    assert(len(image_files) > len(flow_files))

    for i in range(len(flow_files)):
        image_file1 = image_files[i]
        image_file2 = image_files[i+1]
        flow_file = flow_files[i]

        flow_file_name = os.path.split(flow_file)[1]
        flow_file_name = os.path.splitext(flow_file_name)[0]
        output_prefix = os.path.join(args.output_folder, flow_file_name)
        vis_flow_arrows(image_file1, image_file2, flow_file, args.thres, output_prefix)
        print "Finish", output_prefix

if __name__ == "__main__":
    main()