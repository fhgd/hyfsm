from pylab import *

import matplotlib.patches as patches
from matplotlib.patches import (
    inside_circle,
    split_bezier_intersecting_with_closedpath,
    make_wedged_bezier2,
    get_parallels,
    Path,
    _point_along_a_line,
)
from matplotlib.bezier import NonIntersectingPathException


class SimpleAB(patches.ArrowStyle.Simple):
    def transmute(self, path, mutation_size, linewidth):

        x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

        # divide the path into a head and a tail
        head_length = self.head_length * mutation_size
        in_f = inside_circle(x2, y2, head_length)
        arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

        try:
            _, arrow_in = \
                  split_bezier_intersecting_with_closedpath(arrow_path,
                                                            in_f,
                                                            tolerence=0.01)
        except NonIntersectingPathException:
            # if this happens, make a straight line of the head_length
            # long.
            x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
            x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
            arrow_in = [(x0, y0), (x1n, y1n), (x2, y2)]


        out_f = inside_circle(x0, y0, head_length)
        try:
            arrow_out, _ = \
                  split_bezier_intersecting_with_closedpath(arrow_path,
                                                            out_f,
                                                            tolerence=0.01)
        except NonIntersectingPathException:
            # if this happens, make a straight line of the head_length
            # long.
            #~ x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
            #~ x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
            #~ arrow_in = [(x0, y0), (x1n, y1n), (x2, y2)]
            arrow_out = None

        # head
        head_width = self.head_width * mutation_size
        head_left, head_right = make_wedged_bezier2(arrow_in,
                                                    head_width / 2., wm=.5)

        # tail
        if arrow_out is not None:
            tail_width = self.tail_width * mutation_size
            tail_left, tail_right = make_wedged_bezier2(arrow_out[::-1],
                                                        head_width / 2.,
                                                        wm=.5)

            mid = [arrow_in[0], (x1, y1), arrow_out[2]]
            mid_left, mid_right = get_parallels(mid, tail_width / 2.)

            if 0:
                plot([x0, x1, x2], [y0, y1, y2], 'o-', ms=20)

                plot(*zip(*head_left), 'o-')
                plot(*zip(*head_right), 'o-')

                plot(*zip(*arrow_in), '.-')
                plot(*zip(*arrow_out), '.-')

                plot(*zip(*tail_left), 'o-')
                plot(*zip(*tail_right), 'o-')

                plot(*zip(*mid_left), 'o-')
                plot(*zip(*mid_right), 'o-')

            patch_path = [(Path.MOVETO, tail_right[2]),
                          (Path.CURVE3, tail_right[1]),
                          (Path.CURVE3, tail_right[0]),

                          (Path.LINETO, mid_right[2]),
                          (Path.CURVE3, mid_right[1]),
                          (Path.CURVE3, mid_right[0]),

                          (Path.LINETO, head_left[0]),
                          (Path.CURVE3, head_left[1]),
                          (Path.CURVE3, head_left[2]),

                          #~ (Path.CURVE3, head_right[2]),
                          (Path.CURVE3, head_right[1]),
                          (Path.CURVE3, head_right[0]),

                          (Path.LINETO, mid_left[0]),
                          (Path.CURVE3, mid_left[1]),
                          (Path.CURVE3, mid_left[2]),

                          (Path.LINETO, tail_left[0]),
                          (Path.CURVE3, tail_left[1]),
                          (Path.CURVE3, tail_left[2]),

                          (Path.CLOSEPOLY, tail_right[2]),
                          ]
        else:
            patch_path = [(Path.MOVETO, head_right[0]),
                          (Path.CURVE3, head_right[1]),
                          (Path.CURVE3, head_right[2]),
                          (Path.CURVE3, head_left[1]),
                          (Path.CURVE3, head_left[0]),
                          (Path.CLOSEPOLY, head_left[0]),
                          ]
        path = Path([p for c, p in patch_path], [c for c, p in patch_path])
        return path, True


simpleA = patches.ArrowStyle.Simple(head_length=0.05, head_width=0.2)
simpleAB = SimpleAB(head_length=0.05, head_width=0.2)


if __name__ == '__main__':
    p = patches.FancyArrowPatch([0.4, 0.25], [0, 0.25],
        arrowstyle=simpleAB,
        mutation_scale=100,
        fc='orange',
        ec='red',
    )
    gca().add_patch(p)
    show()
