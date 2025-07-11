# extraction of car plate vertices from image path/ name
# example of a string of vertices: -386&473_177&454_154&383_363&402-
def vertices_from_image_path(path: str):
    raw_vertices = path.split('-')

    if len(raw_vertices) <= 4: return [0,0,0,0,0,0,0,0] # error prevention
    else: raw_vertices = raw_vertices[3] # always in 4th position

    raw_vertices = raw_vertices.split('_')
    vertices = []
    for x in raw_vertices: vertices += x.split('&')
    return list(map(int, vertices))
