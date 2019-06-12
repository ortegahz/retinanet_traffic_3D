import cv2
import numpy as np
from math import sqrt
from copy import copy

from dataset_utils.geometry import is_right, line, intersection, distance


def warp_generator(image, bb3d, vp1, vp2, im_h, im_w):
    bb3d = np.array(bb3d, np.float32)
    M, _ = get_transform_matrix(vp1, vp2, image, im_w, im_h)

    # image_t = cv2.warpPerspective(image, M, (im_w, im_h), borderMode=cv2.BORDER_REPLICATE)
    image_t = cv2.warpPerspective(image, M, (im_w, im_h), borderMode=cv2.BORDER_CONSTANT)
    t_bb3d = cv2.perspectiveTransform(np.array([[point] for point in bb3d], np.float32), M)

    xs = [point[0][0] for point in t_bb3d]
    ys = [point[0][1] for point in t_bb3d]

    bb_out = {'x_min': np.amin(xs), 'y_min': np.amin(ys), 'x_max': np.amax(xs), 'y_max': np.amax(ys)}

    front = [0, 1, 4, 5]
    xs = [xs[idx] for idx in front]
    ys = [ys[idx] for idx in front]

    bb_in = {'x_min': np.amin(xs), 'y_min': np.amin(ys), 'x_max': np.amax(xs), 'y_max': np.amax(ys)}


    # image_t = cv2.rectangle(image_t,(bb_out['x_min'],bb_out['y_min']),(bb_out['x_max'],bb_out['y_max']),(255,255,255))
    # image_l = cv2.rectangle(image_t,(bb_in['x_min'],bb_in['y_min']),(bb_in['x_max'],bb_in['y_max']),(255,0,255))
    # image_p = copy(image_t)
    # for point in t_bb3d:
    #     cv2.circle(image_p,tuple(point[0]),5,(255,0,0))
    #
    # cv2.imshow('out',image_t)
    # cv2.imshow('in',image_l)
    # cv2.imshow('all',image_p)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_t, M, bb_in, bb_out


def warp_inference(image, vp1, vp2, im_h, im_w):
    M, IM = get_transform_matrix(vp1, vp2, image, im_w, im_h)
    image_t = cv2.warpPerspective(image, M, (im_w, im_h), borderMode=cv2.BORDER_REPLICATE)
    return image_t, M, IM


def unwarp_inference(image, M, bb_in, bb_out):
    if abs(bb_in['x_max'] - bb_out['x_max'] < abs(bb_out['x_min'] - bb_out['x_min'])):
        # bb_in is in lower right corner
        bb_out_x_offset = bb_out['x_min'] - bb_in['x_min']
    else:
        bb_out_x_offset = bb_out['x_max'] - bb_in['x_max']

    bb_out_y_offset = bb_out['y_min'] - bb_in['y_min']

    bb_out_offset = np.ndarray([bb_out_x_offset, bb_out_y_offset], np.float32)

    bb3d = []
    bb3d.append(bb_in['x_min'], bb_in['y_min'])
    bb3d.append(bb_in['x_max'], bb_in['y_min'])
    bb3d.append(bb_in['x_max'], bb_in['y_max'])
    bb3d.append(bb_in['x_min'], bb_in['y_max'])

    bb3d = np.ndarray(bb3d, np.float32)
    bb3d = np.vstack(bb3d, bb3d + bb_out_offset)

    bb3d_it = cv2.perspectiveTransform(bb3d, cv2.invert(M))

    return bb3d_it


# find appropriate corner points
def find_cornerpts(VP, pts):
    # x = VP[0]
    # y = VP[1]
    # if (x <= 0 and y <= 0):
    #     P1 = 3
    #     P2 = 1
    # elif (0 <= x and x <= w and y <= 0):
    #     P1 = 0
    #     P2 = 1
    # elif (w <= x and y <= 0):
    #     P1 = 0
    #     P2 = 2
    # elif (w <= x and 0 <= y and y <= h):
    #     P1 = 1
    #     P2 = 2
    # elif (w <= x and h <= y):
    #     P1 = 1
    #     P2 = 3
    # elif (0 <= x and x <= w and h <= y):
    #     P1 = 2
    #     P2 = 3
    # elif (x <= 0 and h <= y):
    #     P1 = 2
    #     P2 = 0
    # else:
    #     P1 = 3
    #     P2 = 0


    for P1 in range(len(pts)):
        bad = False
        for idx in range(len(pts)):
            if idx != P1 and is_right(VP, pts[P1], pts[idx]):
                bad = True
                break
        if not bad:
            break

    for P2 in range(len(pts)):
        bad = False
        for idx in range(len(pts)):
            if idx != P2 and not is_right(VP, pts[P2], pts[idx]):
                bad = True
                break
        if not bad:
            break

    #P1 is to the right
    #P2 is to the left
    return P1, P2


def get_pts_from_mask(mask, vp1, vp2):
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(countours[0])
    pts = hull[:,0,:]
    idx1, idx2 = find_cornerpts(vp1, pts)
    idx3, idx4 = find_cornerpts(vp2, pts)
    pts = pts[[idx1, idx2, idx3, idx4]]

    return [pts[0], pts[3], pts[2], pts[1]]

    # epsilon = 0.1 * cv2.arcLength(countours[0], True)
    # approx = cv2.approxPolyDP(countours[0], epsilon, True)
    # while len(approx) != 4:
    #     if len(approx) > 4:
    #         epsilon *= 1.1
    #     else:
    #         epsilon *= 0.9
    #     approx = cv2.approxPolyDP(countours[0], epsilon, True)

    # cv2.drawContours(mask, ret[1][0], 0, (255, 255, 255), 3)
    # mask = cv2.drawContours(mask, countours, 0, (0, 0, 255), 3)
    # cv2.imshow("Hull", mask)
    # cv2.waitKey(0)

    # pts = [[0, 0], [mask.shape[1], 0], [mask.shape[1], mask.shape[0]], [0, mask.shape[0]]]
    # approx = [point[0] for point in approx]
    #
    # return [approx[0], approx[3], approx[2], approx[1]]


def get_transform_matrix_with_criterion(vp1, vp2, mask, im_w, im_h, constraint=0.5, enforce_vp1=True):
    pts = get_pts_from_mask(mask, vp1, vp2)
    # pts =[[0, 0], [mask.shape[1], 0], [mask.shape[1], mask.shape[0]], [0, mask.shape[0]]]
    image = 255 * np.ones([mask.shape[0], mask.shape[1]])

    M, IM = get_transform_matrix(vp1, vp2, mask, im_w, im_h, pts=pts, enforce_vp1=enforce_vp1)
    t_image = cv2.warpPerspective(image, M, (im_w, im_h), borderMode=cv2.BORDER_CONSTANT)

    while cv2.countNonZero(t_image)/(im_w*im_h) < constraint:
        print(cv2.countNonZero(t_image) / (im_w * im_h))
        # g_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
        _, b_image = cv2.threshold(t_image, 177, 255, 0)
        b_image = 255 - b_image


        Mom = cv2.moments(b_image)
        cX = Mom["m10"] / Mom["m00"]
        cY = Mom["m01"] / Mom["m00"]
        print("cX: {} cY:{}".format(cX,cY))

        t_pts = [[im_w, 0], [im_w, im_h], [0, im_h], [0, 0]]
        best = 999999999
        best_idx = 0
        for idx, p in enumerate(t_pts):
            val = (cX - p[0])**2 + (cY - p[1])**2
            if val < best:
                best = val
                best_idx = idx

        mp_t = t_pts[best_idx]
        mp_t[0] += np.sign(im_h/2 - mp_t[0]) * 3
        mp_t[1] += np.sign(im_h/2 - mp_t[1]) * 3
        mp_t = np.array([[mp_t]])
        mp = cv2.perspectiveTransform(mp_t, IM)
        mp = mp[0][0]
        pts[best_idx] = mp

        cv2.imshow("t_image", b_image)
        cv2.waitKey(0)



        # if cX < im_w/2 and cY < im_h/2:
        #     pts[0][0] += 5
        #     pts[0][1] += 5
        #     pts[1][1] += 5
        #     pts[3][0] += 5
        # if cX >= im_w/2 and cY < im_h/2:
        #     pts[0][1] += 5
        #     pts[1][0] -= 5
        #     pts[1][1] += 5
        #     pts[2][0] -= 5
        # if cX >= im_w/2 and cY >= im_h/2:
        #     pts[2][0] -= 5
        #     pts[2][1] -= 5
        #     pts[1][0] -= 5
        #     pts[3][1] -= 5
        # if cX < im_w/2 and cY >= im_h/2:
        #     pts[3][0] += 5
        #     pts[3][1] -= 5
        #     pts[2][1] -= 5
        #     pts[0][0] += 5

        print(pts)
        M, IM = get_transform_matrix(vp1, vp2, mask, im_w, im_h, pts=pts, enforce_vp1=enforce_vp1)
        t_image = cv2.warpPerspective(image, M, (im_w, im_h), borderMode=cv2.BORDER_CONSTANT)

    return M, IM


def get_transform_matrix(vp1, vp2, image, im_w, im_h, pts=None, enforce_vp1=True):
    if pts is None:
        pts = [[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0,image.shape[0]]]

    vp1p1, vp1p2 = find_cornerpts(vp1, pts)
    vp2p1, vp2p2 = find_cornerpts(vp2, pts)


    if (is_right(vp1, pts[vp1p1], vp2) != is_right(vp1, pts[vp1p2], vp2)) or (
            is_right(vp2, pts[vp2p1], vp1) != is_right(vp2, pts[vp2p2], vp1)):
        ### Cut just a bit
        if vp1p1 == vp2p1 or vp1p1 == vp2p2:
            corner = vp1p1
        elif vp1p2 == vp2p1 or vp1p2 == vp2p2:
            corner = vp1p2

        a = intersection(line(vp1,vp2),line(pts[corner],pts[(corner - 1) % 4]))
        b = intersection(line(vp1,vp2),line(pts[corner],pts[(corner + 1) % 4]))
        newcorner = intersection(line(a,b), line([image.shape[1] / 2, image.shape[0] / 2], pts[corner]))
        pts[corner] = np.array(newcorner) + 0.4 * (np.array([image.shape[1] / 2, image.shape[0] / 2]) - np.array(newcorner))

        p = [pts[2][1] / 2, pts[2][0]/ 2]
        lvp = line(vp1, vp2)

        # perpendicular to the line
        a = lvp[1]
        b = -lvp[0]
        # going through p
        c = a * p[0] + b * p[1]
        lp = (a, b, c)
        p2 = intersection(lvp, lp)
        midp = (np.array(p) + np.array(p2))/2
        if midp[0] < p[0] and midp[1] < p[1]:
            corner = 0
        elif midp[0] >= p[0] and midp[1] < p[1]:
            corner = 1
        elif midp[0] >= p[0] and midp[1] >= p[1]:
            corner = 2
        else:
            corner = 3

        a = intersection(line(vp1, vp2), line(pts[corner], pts[(corner - 1) % 4]))
        b = intersection(line(vp1, vp2), line(pts[corner], pts[(corner + 1) % 4]))
        newcorner = intersection(line(a, b), line([image.shape[1] / 2, image.shape[0] / 2], pts[corner]))
        midp = np.array(newcorner) + 0.5 * (np.array([image.shape[1] / 2, image.shape[0] / 2]) - np.array(newcorner))
        pts[corner] = midp

        if vp1p1 == corner or vp1p2 == corner:
            if abs(vp2p1 - corner) == 1:
                vp2p1 = corner
            if abs(vp2p2 - corner) == 1:
                vp2p2 = corner
        elif vp2p1 == corner or vp2p2 == corner:
            if abs(vp1p1 - corner) == 1:
                vp1p1 = corner
            if abs(vp1p2 - corner) == 1:
                vp1p2 = corner

    # right side
    vp1l1 = line(vp1, pts[vp1p1])
    # left side
    vp1l2 = line(vp1, pts[vp1p2])
    # right side
    vp2l1 = line(vp2, pts[vp2p1])
    # left side
    vp2l2 = line(vp2, pts[vp2p2])

    t_dpts = [[0, 0], [0, im_h], [im_w, im_h], [im_w, 0]]

    ipts = []
    ipts.append(intersection(vp1l1, vp2l1))
    ipts.append(intersection(vp1l2, vp2l1))
    ipts.append(intersection(vp1l1, vp2l2))
    ipts.append(intersection(vp1l2, vp2l2))

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for p in ipts:
        image = cv2.circle(image,p,40,(0,0,255),thickness=3)
    cv2.imshow("Mask with pts", image)
    cv2.waitKey(0)

    if enforce_vp1:

        if vp1[1] > im_h:
            t_dpts = [[im_w, 0], [im_w, im_h], [0, im_h], [0, 0]]

        t_ipts = np.zeros((4,2), dtype=np.float32)
        t_pts = np.array(t_dpts, np.float32)

        if ipts[0][1] < ipts[2][1]:
            t_ipts[0, :] = ipts[0]
            t_ipts[1, :] = ipts[2]
        else:
            t_ipts[0, :] = ipts[2]
            t_ipts[1, :] = ipts[0]
        if ipts[1][1] < ipts[3][1]:
            t_ipts[3, :] = ipts[1]
            t_ipts[2, :] = ipts[3]
        else:
            t_ipts[3, :] = ipts[3]
            t_ipts[2, :] = ipts[1]

        return cv2.getPerspectiveTransform(t_ipts, t_pts), cv2.getPerspectiveTransform(t_pts, t_ipts)


    ipts = np.array(ipts, np.float32)

    # dpts = [[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]]
    # perms = itertools.permutations(range(4))
    # min = 50000000
    # for list in perms:
    #     val = reduce((lambda x, y: x + y),
    #                  [sqrt((ipts[idx][0] - dpts[list[idx]][0]) ** 2 + (ipts[idx][1] - dpts[list[idx]][1]) ** 2) for idx
    #                   in range(4)])
    #     if val < min:
    #         res = list
    #         min = val

    x_order = np.argsort(ipts[:, 0])
    set_x_left = set(x_order[:2])
    set_x_right = set(x_order[2:])
    y_order = np.argsort(ipts[:, 1])
    set_y_top = set(y_order[:2])
    set_y_bottom = set(y_order[2:])

    if enforce_vp1:
        vp1_dists = [sqrt((ipts[idx][0] - vp1[0]) ** 2 + (ipts[idx][1] - vp1[1]) ** 2) for idx in range(4)]
        y_order = np.argsort(vp1_dists)
        set_y_top = set(y_order[:2])
        set_y_bottom = set(y_order[2:])

    res = []
    res.append(set_x_left.intersection(set_y_top).pop())
    res.append(set_x_left.intersection(set_y_bottom).pop())
    res.append(set_x_right.intersection(set_y_bottom).pop())
    res.append(set_x_right.intersection(set_y_top).pop())

    # t_pts = np.array([t_dpts[idx] for idx in res], np.float32)
    t_pts = np.array(t_dpts, np.float32)
    t_ipts = ipts[res,:]

    return cv2.getPerspectiveTransform(t_ipts, t_pts), cv2.getPerspectiveTransform(t_pts, t_ipts)



def decode_3dbb(boxes, image, IM, vp0, vp1, vp2, vp0_t, threshold = 0.7, ret_centers= False):
    image_b = copy(image)
    if ret_centers:
        centers = []

    for box in boxes:
        if box[0] < threshold:
            continue
        # print(box)
        xmin = box[-5]
        ymin = box[-4]
        xmax = box[-3]
        ymax = box[-2]
        cy_0 = box[-1] * (ymax - ymin) + ymin
        # cy_0 = ymin

        # cx,cy = intersection(line([xmax,ymin],vp0_t),line([0,cy_0],[1,cy_0]))
        # cx,cy = intersection(line([xmax,ymin],vp0_t),line([cx_0,0],[cx_0,1]))

        bb_t = []
        if vp0_t[0] < xmin:
            cx, cy = intersection(line([xmin, ymin], vp0_t), line([0, cy_0], [1, cy_0]))
            if ret_centers:
                center_x = (xmax + cx) / 2
            bb_t.append([cx, cy])
            bb_t.append([xmax, cy])
            bb_t.append([xmax, ymax])
            bb_t.append([cx, ymax])
            bb_t.append([xmin, ymin])

            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, IM)
            bb_tt = [point[0] for point in bb_tt]

            # image_b = cv2.rectangle(image_b, tuple(bb_tt[4]), tuple(bb_tt[2]), (0, 0, 255), 3)

            bb_tt.append(intersection(line(bb_tt[1], vp0), line(bb_tt[4], vp1)))
            bb_tt.append(intersection(line(bb_tt[2], vp0), line(bb_tt[5], vp2)))
            bb_tt.append(intersection(line(bb_tt[3], vp0), line(bb_tt[6], vp1)))

        elif xmax < vp0_t[0]:
            cx, cy = intersection(line([xmax, ymin], vp0_t), line([0, cy_0], [1, cy_0]))
            bb_t.append([cx, cy])
            bb_t.append([cx, ymax])
            bb_t.append([xmin, ymax])
            bb_t.append([xmin, cy])
            bb_t.append([xmax, ymin])

            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, IM)
            bb_tt = [point[0] for point in bb_tt]

            # image_b = cv2.rectangle(image_b, tuple(bb_tt[4]), tuple(bb_tt[2]), (0, 0, 255), 3)

            bb_tt.append(intersection(line(bb_tt[1], vp0), line(bb_tt[4], vp2)))
            bb_tt.append(intersection(line(bb_tt[2], vp0), line(bb_tt[5], vp1)))
            bb_tt.append(intersection(line(bb_tt[3], vp0), line(bb_tt[6], vp2)))

        else:
            # cx, cy = intersection(line([xmin, cy_0], vp0_t), line([xmin, ymin], [xmax, ymin]))
            cy = cy_0
            cx = xmin
            bb_t.append([cx, cy])
            bb_t.append([xmax, cy])
            bb_t.append([xmax, ymax])
            bb_t.append([cx, ymax])

            tx, ty = intersection(line([cx, cy], vp0_t), line([xmin, ymin], [xmin + 1, ymin]))

            bb_t.append([tx, ymin])

            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, IM)
            bb_tt = [point[0] for point in bb_tt]

            # image_b = cv2.rectangle(image_b, tuple(bb_tt[4]), tuple(bb_tt[2]), (0, 0, 255), 3)

            bb_tt.append(intersection(line(bb_tt[1], vp0), line(bb_tt[4], vp1)))
            bb_tt.append(intersection(line(bb_tt[2], vp0), line(bb_tt[5], vp2)))
            bb_tt.append(intersection(line(bb_tt[3], vp0), line(bb_tt[6], vp1)))


        # plt.figure(figsize=(10, 6))
        # plt.imshow(image)
        # for point in bb_tt:
        #     plt.plot(point[0][0], point[0][1], 'ro')
        # plt.show()
        # plt.wait(0.5)
        if ret_centers:
            center = (bb_tt[0] + bb_tt[2]) / 2
            print(center)
            center_x = center[0]
            image_b = cv2.circle(image_b,(np.round(center_x).astype(int),np.round(bb_tt[2][1]).astype(int)),10,(0, 255, 0))
            centers.append([center_x,bb_tt[2][1]])

        bb_tt = [tuple(point) for point in bb_tt]

        image_b = cv2.line(image_b, bb_tt[0], bb_tt[1], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[1], bb_tt[2], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[2], bb_tt[3], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[3], bb_tt[0], (0, 0, 255), 2)

        image_b = cv2.line(image_b, bb_tt[0], bb_tt[4], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[1], bb_tt[5], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[2], bb_tt[6], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[3], bb_tt[7], (0, 0, 255), 2)

        image_b = cv2.line(image_b, bb_tt[4], bb_tt[5], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[5], bb_tt[6], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[6], bb_tt[7], (0, 0, 255), 2)
        image_b = cv2.line(image_b, bb_tt[7], bb_tt[4], (0, 0, 255), 2)

    if ret_centers:
        return image_b, centers


    return image_b


