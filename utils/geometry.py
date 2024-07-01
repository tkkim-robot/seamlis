from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def custom_merge(geometries):
    # First, perform a union operation
    union = unary_union(geometries)
    
    # If the result is a MultiPolygon, we need to process each polygon
    if isinstance(union, MultiPolygon):
        processed_polygons = [process_polygon(poly, geometries) for poly in union.geoms]
        return MultiPolygon(processed_polygons)
    elif isinstance(union, Polygon):
        return process_polygon(union, geometries)
    else:
        return union  # In case it's neither Polygon nor MultiPolygon

def process_polygon(polygon, geometries):
    # Preserve exterior and interior rings
    exterior = polygon.exterior
    interiors = list(polygon.interiors)
    
    # Check if any of the original geometries create new holes
    for geom in geometries:
        if isinstance(geom, Polygon):
            new_interiors = find_new_interiors(polygon, geom)
            interiors.extend(new_interiors)
        elif isinstance(geom, MultiPolygon):
            for sub_poly in geom.geoms:
                new_interiors = find_new_interiors(polygon, sub_poly)
                interiors.extend(new_interiors)
    
    # Create a new polygon with the original exterior and all interiors
    return Polygon(exterior, interiors)

def find_new_interiors(main_poly, sub_poly):
    # Find parts of sub_poly that create new holes in main_poly
    diff = main_poly.difference(sub_poly)
    if isinstance(diff, Polygon):
        return list(diff.interiors)
    elif isinstance(diff, MultiPolygon):
        return [interior for poly in diff.geoms for interior in poly.interiors]
    else:
        return []