import numpy as np
import cv2


def find_max_enclosed_rect(test_mask):

    dimMask = test_mask
    dimMask[dimMask > 0] = 1
    xR, yR, wR, hR = cv2.boundingRect(dimMask)

    orig_mask_sum = np.sum(test_mask)
    orig_mask_sum = orig_mask_sum.astype('float64')
    # Make 75% mask
    rec_image = np.zeros((test_mask.shape), 'uint8')
    x1 = xR+int(np.round(wR/8))
    x2 = xR+int(np.round(0.875*wR))
    y1 = yR+int(np.round(hR/8))
    y2 = yR+int(np.round(0.875*hR))

    coords_fill = [x1, y1, x2, y2]
    rec_image = cv2.rectangle(rec_image, (x1, y1), (x2, y2), 1, thickness=-1)

    # Grow mask until it intersects with stage area
    loop_count = 0
    finished_coords = []

    while len(finished_coords) < 4:

        # Cycle through coordinate values
        loop_count += 1
        coord_idx = loop_count % 4
        coord_val = coords_fill[coord_idx]

        # expand coordinates to make bigger ROI

        if coord_idx < 2:
            new_coord_val = coord_val - 1
        else:
            new_coord_val = coord_val + 1

        # Check that resultant mask doesn't add zeroes
        coords_fill[coord_idx] = new_coord_val

        rec_image_temp = cv2.rectangle(
              rec_image,
              (coords_fill[0], coords_fill[1]),
              (coords_fill[2], coords_fill[3]),
              1,
              thickness=-1)

        new_mask_sum = np.sum(rec_image_temp)

        mask_diff = test_mask-rec_image_temp
        mask_diff_sum = np.sum(mask_diff)
        mask_diff_sum = mask_diff_sum.astype('float64')
        diff = orig_mask_sum - mask_diff_sum

        if new_mask_sum > diff:
            coords_fill[coord_idx] = coord_val
            finished_coords = np.append(finished_coords, int(coord_idx))
            finished_coords = np.unique(finished_coords)

    return rec_image_temp, coords_fill, loop_count


def find_bounding_rectangle(img_mask):

    return cv2.boundingRect(img_mask)
