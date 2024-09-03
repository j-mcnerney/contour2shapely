import numpy as np
from contour2shapely import (
    multi_polygon_from_contourset,
    plot_surface_and_return_contours,
)

X_RANGE = (-10, 10)
Y_RANGE = (-10, 10)
N_POINTS = 3000
TOL = 1e-2


def main():
    Xmat, Ymat, Zmat = complex_bump_scenario()

    contour_level = 1
    contourset = plot_surface_and_return_contours(Xmat, Ymat, Zmat, contour_level)

    mp = multi_polygon_from_contourset(contourset)


def complex_bump_scenario():
    # Upper left
    Xmat, Ymat, Zmat = plateau(xc=-5, yc=3, height=5, radius=4)
    _, _, Zmat = plateau(xc=-5, yc=5, height=-5, radius=1, Z0=Zmat)
    _, _, Zmat = plateau(xc=-5, yc=1, height=-5, radius=1, Z0=Zmat)

    # Upper right
    _, _, Zmat = plateau(xc=5, yc=5, height=5, radius=4, Z0=Zmat)
    _, _, Zmat = plateau(xc=5, yc=5, height=-5, radius=3, Z0=Zmat)
    _, _, Zmat = plateau(xc=5, yc=5, height=5, radius=1, Z0=Zmat)

    # Lower middle
    _, _, Zmat = plateau(xc=3, yc=-5, height=5, radius=4, Z0=Zmat)
    _, _, Zmat = plateau(xc=3, yc=-5, height=-5, radius=3, Z0=Zmat)
    _, _, Zmat = plateau(xc=3, yc=-5, height=5, radius=2, Z0=Zmat)
    _, _, Zmat = plateau(xc=3, yc=-5, height=-5, radius=1, Z0=Zmat)

    return Xmat, Ymat, Zmat


def plateau(xc=0, yc=0, height=1, radius=1, Z0=None):
    # Create a grid
    x = np.linspace(*X_RANGE, N_POINTS)
    y = np.linspace(*Y_RANGE, N_POINTS)
    Xmat, Ymat = np.meshgrid(x, y)

    # Create a bump
    R_squared = (Xmat - xc) ** 2 + (Ymat - yc) ** 2
    Zmat = np.zeros_like(R_squared)
    Zmat[R_squared <= radius**2] = height

    # If Z0 was provided, add to the current Zmat
    if Z0 is not None:
        Zmat = Zmat + Z0
    return Xmat, Ymat, Zmat


def test_complex_bump_scenario():
    # Setup
    Xmat, Ymat, Zmat = complex_bump_scenario()
    contourset = plot_surface_and_return_contours(Xmat, Ymat, Zmat, contour_level=1)
    mp = multi_polygon_from_contourset(contourset)

    # Compute area
    area = mp.area

    expected_area = np.pi * (4**2 + 4**2 + 1**2 + 4**2 + 2**2) - np.pi * (
        1**2 + 1**2 + 3**2 + 3**2 + 1**2
    )
    assert np.isclose(
        area, expected_area, rtol=TOL
    ), f"Expected {expected_area}, got {area}"


if __name__ == "__main__":
    main()
    test_complex_bump_scenario()
