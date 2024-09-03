import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from matplotlib.contour import ContourSet
import matplotlib.pyplot as plt


def plot_surface_and_return_contours(Xmat, Ymat, Zmat, contour_level):
    # Setup figure
    fig = plt.figure(num=1, clear=True, figsize=(4, 7))

    # ========#
    # Plot 3d
    ax = fig.add_subplot(211, projection="3d")
    ax.plot_surface(Xmat, Ymat, Zmat)

    # Refine
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Bumpy 3D Surface")
    ax.set_zlim(0, 6)
    ax.set_aspect("equal")

    # ========#
    # Plot 2d
    ax = fig.add_subplot(212)
    contour_set = ax.contourf(
        Xmat,
        Ymat,
        Zmat,
        levels=[contour_level, np.inf],
        colors=["orange", "red"],
    )

    # Refine
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Contours")
    ax.set_aspect("equal")

    plt.show()

    return contour_set


def multi_polygon_from_contourset(contourset: ContourSet):
    # Extract segments from contourset
    segments = contourset.allsegs[0]

    # Determine containment relationships between segments
    polygons = [Polygon(segment) for segment in segments]
    containments = np.zeros((len(polygons), len(polygons)), dtype=bool)
    for i, polygon_i in enumerate(polygons):
        for j, polygon_j in enumerate(polygons):
            if i == j:
                continue
            if polygon_j.contains(polygon_i):
                containments[i, j] = True

    # Compute "order" of each polygon
    # - order: # of other polygons that contain it
    # - zero order connotes an outermost shell
    # - odd orders are holes
    # - even orders are shells
    polygon_orders = containments.sum(axis=1)

    # Gather the shells (even-order polygons)
    shell_polygons = [
        polygons[i] for i, order in enumerate(polygon_orders) if order % 2 == 0
    ]

    # Create polygons with holes
    polygons_with_holes = []
    for s in shell_polygons:
        # Gather holes of this shell
        shell_order = polygon_orders[polygons.index(s)]
        hole_polygons = [
            polygons[i]
            for i, polygon_order in enumerate(polygon_orders)
            if s.contains(polygons[i]) and polygon_order == shell_order + 1
        ]

        # Create a Polygon with holes
        shell = s.exterior
        holes = [h.exterior for h in hole_polygons]
        polygons_with_holes.append(Polygon(shell, holes))

    # Create MultiPolygon
    return MultiPolygon(polygons_with_holes)
