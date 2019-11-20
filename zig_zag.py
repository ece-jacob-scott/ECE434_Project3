def zig_zag(image):
    [height, width] = image.shape
    intermediate_map = [[] for i in range(height * width + 1)]
    arr = []

    for i in range(height):
        for j in range(width):
            s = i + j
            if s % 2 is 0:
                intermediate_map[s].insert(0, image[i, j])
                continue
            intermediate_map[s].append(image[i, j])

    for i in intermediate_map:
        for j in i:
            arr.append(j)
    return arr


# TODO: implement unzig_zag function
def unzig_zag(arr):
    raise NotImplementedError()
