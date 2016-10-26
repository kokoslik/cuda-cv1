import numpy as np
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_image', type=str, help='input image path', default='img.jpg')
    parser.add_argument('--out_file', type=str, help='output text file path', default='img.txt')

    args = parser.parse_args()

    image_name = args.in_image
    text_file_name = args.out_file

    img = cv2.imread(image_name,0)
    np.savetxt(text_file_name, fmt='%d', X=img, newline='\n')

if __name__ == "__main__":
    main()
