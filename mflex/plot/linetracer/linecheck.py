import numpy as np


def fieldlinecheck(fieldline, xmin, xmax, ymin, ymax):
    split = False
    isplit = 0

    Lx = abs(xmax - xmin)
    Ly = abs(ymax - ymin)

    fieldline_x = np.zeros(len(fieldline))
    fieldline_y = np.zeros(len(fieldline))

    fieldline_x[:] = fieldline[:, 0]
    fieldline_y[:] = fieldline[:, 1]

    fieldlines = list()

    for i in range(len(fieldline) - 2):
        if abs(fieldline_x[i] - fieldline_x[i + 1]) < 0.99 * Lx:
            split = True
            isplit = i
            break
        if abs(fieldline_y[i] - fieldline_y[i + 1]) < 0.99 * Ly:
            split = True
            isplit = i
            break

    if split == True:
        fieldline_minus = fieldline[0:isplit, :]
        fieldline_plus = fieldline[isplit + 1 : len(fieldline) - 1, :]
        fieldlines.append(fieldline_minus)
        fieldlines.append(fieldline_plus)
    elif split == False:
        fieldlines.append(fieldline)

    return fieldlines
