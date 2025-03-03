import numpy as np
import math
import cv2 as cv

from scipy import ndimage

def fix_contrast(img, mask):
    hist = cv.calcHist([img], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
    hist = hist.T
    hist = hist[0]
    contrast_pixel_thresh = sum(hist)/750
    for i in range(0, 256):
        if hist[i] > 0:
            minimum = i
            break
    for i in range(0, 256):
        if hist[255 - i] > contrast_pixel_thresh:
            maximum = 256 - i - 1
            break
    img_e = img.copy()
    img_e = img_e.astype(np.float64)
    h, w = img_e.shape
    for i in range(0, h):
        for x in range(0, w):
            if img_e[i, x] < minimum:
                img_e[i, x] = minimum
            img_e[i, x] = (img_e[i, x] - minimum) / (maximum - minimum)
            if img_e[i, x] >= 1:
                img_e[i, x] = 1
            img_e[i, x] = img_e[i, x] * 255
            img_e[i, x] = math.floor(img_e[i, x])
    img_e = img_e.astype(np.uint8)
    img = img_e
    return img



def masking(img, padding_percent, discard_width_percent):
    black_corners = True
    w, h = img.shape
    shorter_img_edge = int(min(w, h)/2)
    longer_img_edge = int(max(w, h)/2)
    padding = int(shorter_img_edge * padding_percent)
    discard_width = int(shorter_img_edge * discard_width_percent)
    radius = int((math.sqrt(h * h + w * w)) / 2)

    while black_corners:
        radius = radius - padding
        if radius <= longer_img_edge:
            radius = longer_img_edge
            mask = mask_with_corners(img, radius)
            break
        else:
            mask = mask_with_corners(img, radius)
            black_corners = check_corners(img, mask)

    discard_mask = mask_borders(mask, radius, discard_width)

    return mask, discard_mask

def otsu_based_binarisation(hist):
    Inter_variance = -1
    O_thresh = 0
    O_Inter = 0

    for thresh in range(1, 256):
        pixels = sum(hist[0])
        pixels_b = sum(hist[0, range(0, thresh)])
        pixels_f = sum(hist[0, range(thresh, 256)])
        Wb = pixels_b / pixels
        Wf = pixels_f / pixels

        if pixels_b == 0 or pixels_f == 0:
            Inter_variance = 0

        else:
            Mean = hist[0].copy()
            for i in range(0, 256):
                Mean[i] = Mean[i] * i
            mean_b = sum(Mean[0:thresh]) / pixels_b
            mean_f = sum(Mean[thresh:256]) / pixels_f

            Variance = hist[0].copy()
            for i in range(0, 256):
                if i < thresh:
                    Variance[i] = ((i - mean_b) ** 2) * Variance[i]
                else:
                    Variance[i] = ((i - mean_f) ** 2) * Variance[i]
            variance_b = sum(Variance[0:thresh]) / pixels_b
            variance_f = sum(Variance[thresh:256]) / pixels_f

            Inter_variance = variance_b * Wb + variance_f * Wf

        if Inter_variance > 0 or O_Inter == 0.0:
            if O_Inter > Inter_variance or O_Inter == 0:
                O_Inter = Inter_variance
                O_thresh = thresh - 1

    return O_thresh


def denoising(img, img_b, discard_mask):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img_b, connectivity=8)
    size = stats[1:, -1]
    indexes_from_biggestVal = np.flip(np.argsort(size)).astype(np.int32)
    img_b = np.array(img_b)

    for i in range(0,nb_components):
        img_b[:] = False
        img_b[output == indexes_from_biggestVal[i] + 1] = True
        if discard_mask.sum() == 0:
            prawdopodobienstwo_brzegu = 0.0
        else:
            porownanie_brzegi = np.logical_and(img_b[:,:], discard_mask[:,:])
            prawdopodobienstwo_brzegu = porownanie_brzegi.sum() / discard_mask.sum()
        if prawdopodobienstwo_brzegu == 0.0:
            break

    j = indexes_from_biggestVal[i]+1
    centroid_center = [stats[j][0]+int(stats[j][2]/2), stats[j][1]+int(stats[j][3]/2)]
    centroid_size = stats[j,-1]
    img_b = ndimage.binary_fill_holes(img_b).astype(bool)
    img_b = ndimage.morphology.binary_dilation(img_b, iterations=8)
    img_masked = img * img_b
    return img_masked, centroid_center, centroid_size




def mask_with_corners(img, radius):
    img = 255 - img
    h, w = img.shape

    center = [int(h / 2), int(w / 2)]
    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius

    return mask


def check_corners(img, inner_mask):
    outer_mask = np.full(inner_mask.shape, False)
    outer_mask[inner_mask==False] = True
    corners_masked = np.uint64(img * outer_mask)
    corner_pixel_mean = sum(sum(corners_masked))/sum(sum(outer_mask))
    if corner_pixel_mean < 110:
        black_corners = True
    else:
        black_corners = False

    return black_corners


def mask_borders(mask, mask_radius, discard_width):
    h, w = mask.shape

    center = [int(h / 2), int(w / 2)]
    radius = mask_radius - discard_width
    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    circle = dist_from_center <= radius

    canvas = np.full(mask.shape, False)
    canvas[:int(discard_width), :] = True
    canvas[int(h - discard_width):, :] = True
    canvas[:, :int(discard_width)] = True
    canvas[:, int(w - discard_width):] = True
    rectangle = np.invert(canvas)
    mask2 = circle & rectangle

    xor_mask = np.logical_xor(mask, mask2)
    xor_mask = ndimage.binary_dilation(xor_mask, iterations=1)
    return xor_mask



def orb_match(des1, des2):
    bf = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    o1, k1 = des1.shape
    o2, k2 = des2.shape
    o = (o1 + o2)/2
    percent = (len(matches) / o) * 100
    percent = round(percent, 2)
    return len(matches), percent





def preprocess(image):
    im = image.copy()
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    grayScale = im.copy()
    kernel = cv.getStructuringElement(1, (17, 17))
    blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)
    ret, thresh = cv.threshold(blackhat, 10, 255, cv.THRESH_BINARY)
    im = cv.inpaint(im, thresh, 1, cv.INPAINT_NS)

    inner_mask, discard_mask = masking(im, 0.05, 0.05)
    im_m = (255 - im) * inner_mask
    im = fix_contrast(im, im_m)

    im = 255 - im

    hist_f = cv.calcHist([im], channels=[0], mask=im, histSize=[256], ranges=[0, 256])
    hist_f = hist_f.T
    thresh_f = otsu_based_binarisation(hist_f)
    img_b = np.uint8(im > thresh_f)

    img_lap = ndimage.laplace(img_b)
    img_lap = ndimage.binary_dilation(img_lap)
    img_b[img_lap] = 0

    im_m, _, _ = denoising(im, img_b, discard_mask)
    return im, im_m

