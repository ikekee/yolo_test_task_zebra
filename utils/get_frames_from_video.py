import argparse
from pathlib import Path
import sys

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.path_lib import create_path_if_not_exists


def main(video_path: Path, output_dir: Path, scaling_factor: float):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        resized_frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
        cv2.imwrite(output_dir/f"{video_path.stem}_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg",
                    resized_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    create_path_if_not_exists(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        '--video_path',
        type=Path,
        required=True
    )
    parser.add_argument(
        "-o",
        '--output_dir',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-s',
        '--scaling_factor',
        type=float,
        default=0.2,
        required=False
    )
    args = parser.parse_args()
    main(args.video_path, args.output_dir, args.scaling_factor)