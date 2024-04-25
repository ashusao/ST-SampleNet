from haversine import Unit, inverse_haversine, Direction
from shapely.geometry import Point, Polygon


def get_grid_index(grids, w, h, lat, lon):

    point = Point(lon, lat)
    idx = -1
    for i, grid in grids.items():
        if point.within(grid):
            idx = i
            break

    if idx == -1:
        return idx

    return (h - int(idx / w)) - 1, int(idx % w)


def bounding_box_to_grid(min_lat, min_lon, max_lat, max_lon, cell_size=1000):

    grid = {}

    south_west = (min_lat, min_lon)
    north_west = inverse_haversine(south_west, cell_size, Direction.NORTH, unit=Unit.METERS)
    south_east = inverse_haversine(south_west, cell_size, Direction.EAST, unit=Unit.METERS)
    north_east = (north_west[0], south_east[1])
    cell = [south_west, north_west, north_east, south_east, south_west]

    while cell[0][0] < max_lat:
        row_start = cell

        while cell[0][1] < max_lon:
            grid[len(grid)] = Polygon([[x[1], x[0]] for x in cell])
            # north_west = last north_east
            north_west = cell[2]
            # south_west = last south_east
            south_west = cell[3]

            north_east = inverse_haversine(north_west, cell_size, Direction.EAST, unit=Unit.METERS)
            south_east = inverse_haversine(south_west, cell_size, Direction.EAST, unit=Unit.METERS)
            cell = [south_west, north_west, north_east, south_east, south_west]

        # south_west = last north_west
        south_west = row_start[1]
        # south_east = last north_east
        south_east = row_start[2]
        north_west = inverse_haversine(south_west, cell_size, Direction.NORTH, unit=Unit.METERS)
        north_east = inverse_haversine(south_east, cell_size, Direction.NORTH, unit=Unit.METERS)
        cell = [south_west, north_west, north_east, south_east, south_west]

    return grid