import img.core.search as srch
import img.util.gcs as gcs
import img.core.cloud as cld
from typing import List
from img.util.basic import BlobNotFoundError
from pathlib import Path
from img.engine.defect import DefectProcessor
import img.core.label as lbl
import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from img.proto.imglabel_pb2 import BoundingBox
import copy
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import TimeoutError


DEST_DIR = Path.cwd() / "final-project/fa-tears"
DEST_DIR = Path.cwd() / "final-project/fa-tear-test"


@dataclass
class Point:
    x: float
    y: float

    def scale(self, width_ar, height_ar):
        self.x = int(np.round(self.x * width_ar))
        self.y = int(np.round(self.y * height_ar))
        return self

    def normalize(self, img_width: int, img_height: int):
        """ Divides the points by the given width and height of the image

            Args:
                img_width: The width of the image containing this point
                img_height: The height of the image containing this point
        """
        self.x = self.x / img_width
        self.y = self.y / img_height
        return self

    def round(self, precision: int = 2, round_up: bool = True):
        """ Rounds the coordinates to the given precision

            Args:
                precision(int): Number of decimal places to round to
                round_up(bool): Whether or not we are rounding up vs down
        """
        rounding_fn = np.ceil if round_up else np.round

        # np.ceil/floor round to the nearest integer, so we first multiply by
        # 10 ** precision, then round to nearest integer, then divide by 10 **
        # precision so that we round to the correct decimal place
        self.x = rounding_fn(self.x * (10 ** precision)) / (10 ** precision)
        self.y = rounding_fn(self.y * (10 ** precision)) / (10 ** precision)
        return self


@dataclass
class BTLabel:
    name: str
    upper_left: Point
    lower_right: Point

    def contains(self, pt: Point):
        return (
            pt.x > self.upper_left.x and
            pt.y > self.upper_left.y and
            pt.x < self.lower_right.x and
            pt.y < self.lower_right.y
        )

    def scale(self, width_ar: float, height_ar: float):
        """ Given new aspect ratios, scale the current BTLabel
        """
        self.upper_left.scale(width_ar, height_ar)
        self.lower_right.scale(width_ar, height_ar)
        return self

    @staticmethod
    def from_bounding_box(
            name: str, bbox: BoundingBox):
        """ Given a label name and a bounding box, this function returns a
            label ready to be written to a .csv file

            Args:
                name(str): Name of the labeled region (ie: 'dark-spot')
                img_width(int): Width of the image the bounding box is in
                img_height(int): Height of the image the bounding box is in
                bbox(BoundingBox): The bounding box to be converted into
                    a GoogleVisionLabel
        """
        # We round down the upper left point and round up the lower right
        # point to maximize the size of the label
        img_width = bbox.dimensions.width
        img_height = bbox.dimensions.height
        upper_left = Point(
                int(bbox.upper_left.x), int(bbox.upper_left.y)
                ).normalize(img_width, img_height)
        lower_right = Point(
                int(bbox.lower_right.x), int(bbox.lower_right.y)
                ).normalize(img_width, img_height)

        return BTLabel(name, upper_left, lower_right)


def load_images():

    label = "ricky-tear"
    images = srch.search_images(
            sample_prefix="CSC070AE", label_prefix=label, version=2)
    images.extend(srch.search_images(
        sample_prefix="CSC070EZ", label_prefix=label, version=2))
    images.extend(srch.search_images(
        sample_prefix="CSC070FC", label_prefix=label, version=2))

    dest_dir = DEST_DIR

    if not dest_dir.is_dir():
        os.makedirs(dest_dir)

    for cloud_image in images:

        bucket = cloud_image.bucket
        f_path = Path(cloud_image.f_name)

        f_name = "_".join(f_path.stem.split("_")[:-1]) + ".png"
        cloud_path = Path("archive") / f_name

        img_p, _ = gcs.copy_from_cloud(bucket, cloud_path, dest_dir)
        rename_image(img_p)

        # split_str = "_Trans"
        # if split_str not in str(cloud_image.f_name):
        #     split_str = "_1"

        # parts = f_name.split(split_str)

        # trans_joiner = "_Trans" if split_str == "_Ref" else "_1"
        # trans_f_name = Path(parts[0] + trans_joiner + parts[1])

        # cloud_path = Path("archive") / trans_f_name
        # try:
        #     img_p, _ = gcs.copy_from_cloud(bucket, cloud_path, dest_dir)
        #     rename_image(img_p)
        # except BlobNotFoundError:
        #     breakpoint()

    return images


def load_local_images(local_dir: Path):

    image_paths = local_dir.rglob("*.png")


def rename_image(image_p: Path):

    name = image_p.stem
    if name[-2:] == "_0":
        renamed = f"{name[:-2]}_Ref.png"
        os.rename(image_p, image_p.parent / f"{renamed}")

    if name[-2:] == "_1":
        renamed = f"{name[:-2]}_Trans.png"
        os.rename(image_p, image_p.parent / f"{renamed}")


def find_possible_defects(src_dir):

    images = src_dir.glob("*.png")

    for image_path in images:

        rename_image(image_path)
        # For each image, we want to load it and run it through the defect proc
        # We need to modify the run function of that processor a bit
        print(f"Processing {image_path}")
        proc = DefectProcessor()

        proc.run(image_path)


def label_samples(cloud_images: List[cld.CloudImage]):

    executor = ProcessPoolExecutor()
    TO_SKIP = [5]
    n = len(cloud_images)
    for i, cloud_img in enumerate(cloud_images):

        identifier = cloud_img.identifier
        print(f"Processing sample {i} of {n}: {identifier}")
        if(i in TO_SKIP):
            continue

        images = DEST_DIR.glob(f"{identifier.upper()}*.png")

        # We need to load the labels for this image
        key = cloud_img.row_key
        labels = lbl.get_labels_for_image(key, version=2)
        bt_labels = []

        for label in labels.labels:
            if label.name == "ricky-tear":
                for box in label.boxes:
                    bt_labels.append(
                            BTLabel.from_bounding_box(label.name, box))

        for image_p in images:
            if "labeled" in image_p.stem:
                continue
            bt_copy = copy.deepcopy(bt_labels)
            # process_image(image_p, bt_copy)
            try:
                executor.submit(
                        process_image, image_p, bt_copy).result(timeout=300)
            except TimeoutError:
                print(f"Process image timed out for image {image_p.stem}")

        print("Finished processing")


def process_image(image_p: Path, bt_labels: List[BTLabel]):

    proc = DefectProcessor()
    proc.run(image_p)

    # We want to read the image in
    img = cv2.imread(str(image_p))
    # As well as the defect data
    csv_path = image_p.parent / f"{image_p.stem}_defects.csv"

    img_width, img_height = img.shape[1], img.shape[0]

    defect_data = pd.read_csv(csv_path)
    # Scale our labels up to the original size
    for bt_label in bt_labels:
        bt_label.scale(img_width, img_height)
        cv2.rectangle(
            img,
            (bt_label.upper_left.x, bt_label.upper_left.y),
            (bt_label.lower_right.x, bt_label.lower_right.y),
            (0, 0, 255),
            thickness=2
        )

    labeled_data = []
    # Now we have a pandas dataframe, as well as a label
    for i, row in defect_data.iterrows():
        upper_left = Point(int(row.left), int(row.top))
        lower_right = Point(
                int(row.left + row.width), int(row.top + row.height))

        color = (0, 255, 0)
        is_tear = False
        for bt_label in bt_labels:
            # We want to check if this defect label is inside of the bt label
            # Then we will consider it an fa-tear
            if (bt_label.contains(upper_left) and
               bt_label.contains(lower_right)):
                is_tear = True
                color = (0, 0, 255)

        row = row.append(pd.Series([is_tear], ["fa_tear"]))
        row = row.drop(labels=["Unnamed: 0"])
        labeled_data.append(row)

        cv2.rectangle(
            img,
            (upper_left.x, upper_left.y),
            (lower_right.x, lower_right.y),
            color,
            thickness=2
        )
        cv2.putText(
                img,
                f"d-{i}",
                (upper_left.x, upper_left.y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                thickness=1,
                color=color
        )

    labeled_path = image_p.parent / f"{image_p.stem}_labeled.png"
    cv2.imwrite(str(labeled_path), img)

    labeled_df = pd.DataFrame()
    labeled_df = labeled_df.append(labeled_data, ignore_index=True)
    labeled_df.to_csv(csv_path)


if __name__ == "__main__":
    cld_images = load_images()
    find_possible_defects(Path.cwd() / "final-project/fa-tear-test")
    label_samples(cld_images)
