import csv
from pathlib import Path
import numpy as np
import os
import cv2
from img.util.masks import find_sample_mask, keep_largest_object
from dataclasses import dataclass
from fp.utils import find_max_enclosed_rect
import matplotlib.pyplot as plt
import time
import copy

SRC_DIR_TRAIN = Path.cwd() / "final-project/fa-tears"
SRC_DIR_TEST = Path.cwd() / "final-project/fa-tear-test"

DST_DIR_TRAIN = Path(
      "/mnt/c/Users/RRG01/OneDrive - Quantumscape Corporation/fa-tears/train")
DST_DIR_TEST = Path(
      "/mnt/c/Users/RRG01/OneDrive - Quantumscape Corporation/fa-tears/test")

NUM_CROPS = 2
CROP_WIDTH = 200
CROP_HEIGHT = 200


@dataclass
class Defect:

    label_idx: int
    min_int: int
    max_int: int
    std_int: float
    area: int
    left: int
    top: int
    width: int
    height: int
    aspect_ratio: float
    diam_equivalent_um: float
    fa_tear: bool

    def shift(self, up: int, left: int):
        self.top -= int(up)
        self.left -= int(left)

    @staticmethod
    def from_csv_row(csv_row):
        return Defect(
            int(float(csv_row["label_idx"])),
            int(float(csv_row["min_int"])),
            int(float(csv_row["max_int"])),
            float(csv_row["std_int"]),
            int(float(csv_row["area"])),
            int(float(csv_row["left"])),
            int(float(csv_row["top"])),
            int(float(csv_row["width"])),
            int(float(csv_row["height"])),
            float(csv_row["aspect_ratio"]),
            float(csv_row["diam_equivalent_um"]),
            bool(int(float(csv_row["fa_tear"]))),
        )


def load_defect_data():

    csv_paths = SRC_DIR_TRAIN.glob("*Ref_defects.csv")

    data = []
    for csv_p in csv_paths:
        file_data = []
        with csv_p.open("r") as csvfile:
            csv_reader = csv.DictReader(csvfile)

            for row in csv_reader:
                file_data.append(np.asarray([
                    float(row["mean_int"]),            # 1
                    float(row["mean_int_R"]),          # 2
                    float(row["mean_int_B"]),          # 3
                    float(row["mean_int_G"]),          # 4
                    float(row["min_int"]),             # 5
                    float(row["max_int"]),             # 6
                    float(row["std_int"]),             # 7
                    float(row["area"]),                # 8
                    float(row["aspect_ratio"]),        # 9
                    float(row["diam_equivalent_um"]),  # 10
                    float(row["entropy_def_img"]),     # 11
                    float(row["median_def_img"]),      # 12
                    float(row["dispersion_def_img"]),  # 13
                    float(row["fa_tear"])
                ]))
        file_data = np.asarray(file_data)
        data.append(file_data)

    full_data = None
    for img_data in data:
        if full_data is None:
            full_data = img_data
        else:
            full_data = np.vstack((full_data, img_data))

    return full_data


def write_defect_images(src_dir: Path, dest_dir: Path):

    csv_paths = src_dir.glob("*Trans_defects.csv")

    data = []

    true_dir = dest_dir / f"fa-tear"
    if not true_dir.is_dir():
        os.makedirs(true_dir)

    false_dir = dest_dir / f"no-tear"
    if not false_dir.is_dir():
        os.makedirs(false_dir)

    for csv_p in csv_paths:
        file_data = []
        img_fname = "_".join(csv_p.stem.split("_")[:-1])
        img_p = csv_p.parent / f"{img_fname}.png"
        full_img = cv2.imread(str(img_p))

        defects = []
        with csv_p.open("r") as csvfile:
            csv_reader = csv.DictReader(csvfile)

            for row in csv_reader:
                defects.append(Defect.from_csv_row(row))

        # Basically, we want to crop the image for this row
        defect_crops = make_random_crops(full_img, defects)
        for defect, crops in defect_crops:
            for i, crop in enumerate(crops):
                idx = defect.label_idx
                tear = defect.fa_tear
                dst = f"{img_fname}_{idx}-{i+1}.png"
                dst_p = true_dir / dst if tear else false_dir / dst
                cv2.imwrite(str(dst_p), crop)

        file_data = np.asarray(file_data)
        data.append(file_data)


def make_random_crops(image, defects):

    # Create image mask
    img_grey = (
        0.114 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.299 * image[:, :, 2]
    )
    img_mask = find_sample_mask(img_grey)
    img_mask = keep_largest_object(img_mask)
    # original_mask = copy.deepcopy(img_mask)
    img_mask = cv2.resize(
            img_mask, (img_mask.shape[1] // 2, img_mask.shape[0] // 2))

    print("Finding max enclosed rectangle")
    # Find active area
    start = time.time()
    rec_mask, rect_coords, iterations = find_max_enclosed_rect(img_mask)
    end = time.time()
    print(f"Time it took to finish: {end - start}")
    left, top, right, bottom = rect_coords
    left *= 2
    top *= 2
    right *= 2
    bottom *= 2
    cropped_image = image[top:bottom, left:right]

    height, width = cropped_image.shape[:2]
    shift_up, shift_left = top, left
    # canvas = original_mask
    # canvas[original_mask > 0] = 1
    # canvas[top:bottom, left:right] += 1

    # for defect in defects:
    #     cv2.rectangle(
    #         canvas,
    #         (defect.left, defect.top),
    #         (defect.left + defect.width, defect.top + defect.height),
    #         3,
    #         thickness=2
    #     )
    # ratio = canvas.shape[1] / canvas.shape[0]
    # plt.figure(figsize=(20*ratio, 20))
    # plt.imshow(canvas)
    # plt.savefig("defects.png")
    # cv2.imwrite("defects.png", canvas)
    # shaved_mask = shave_mask(img_mask, NUM_EROSIONS)
    # left, top, width, height = cv2.bounding_rect(shaved_mask)

    # Combine overlapping labels into single crops

    defect_crops = []
    for i in range(len(defects)):

        crop_defect = copy.deepcopy(defects[i])
        crop_defect.shift(shift_up, shift_left)

        top = crop_defect.top
        bottom = crop_defect.top + crop_defect.height
        left = crop_defect.left
        right = crop_defect.left + crop_defect.width

        if (top < 0 or bottom > height or left < 0 or right > width):
            print("Defect out of range: ",
                  f"H={height} W={width} ",
                  f"Coords: ({top},{left}), ({bottom},{right}) ",
                  f"fa-tear={crop_defect.fa_tear}")
            continue

        crops = []  # list of cropped images for defect
        for c in range(NUM_CROPS):

            # Make sure top of crop is in frame
            crop_top = np.amax(
                    [top - np.random.randint(CROP_HEIGHT - bottom + top), 0])
            # Make sure bottom of crop is in frame
            crop_top = np.amin([crop_top, height - CROP_HEIGHT])

            crop_bottom = crop_top + CROP_HEIGHT

            crop_left = np.amax(
                    [left - np.random.randint(CROP_WIDTH - right + left), 0])
            # Make sure right of crop is in frame
            crop_left = np.amin([crop_left, width - CROP_WIDTH])
            crop_right = crop_left + CROP_WIDTH

            crop = cropped_image[crop_top:crop_bottom, crop_left:crop_right]
            if crop.shape[0] != CROP_HEIGHT or crop.shape[1] != CROP_WIDTH:
                print("ERROR! Invalid Crop Size")
            crops.append(crop)

        defect_crops.append((crop_defect, crops))

    return defect_crops


if __name__ == "__main__":
    write_defect_images(SRC_DIR_TRAIN, DST_DIR_TRAIN)
    write_defect_images(SRC_DIR_TEST, DST_DIR_TEST)
