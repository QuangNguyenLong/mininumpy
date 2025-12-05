from PIL import Image
import mininumpy as mnp


def read_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    data = list(img.getdata())  # [(R,G,B), ...]

    # Separate channels
    R = [p[0] for p in data]
    G = [p[1] for p in data]
    B = [p[2] for p in data]

    # Concatenate in channel-major order
    flat = R + G + B

    # Now shape is (3, h, w)
    return mnp.Array(flat, shape=(3, h, w))


def save_image(array, path):
    if not isinstance(array, mnp.Array):
        raise TypeError("Expected an Array or np.array instance.")
    if array.shape[0] != 3:
        raise ValueError(f"Expected shape (3, h, w), got {array.shape}")

    c, h, w = array.shape
    data = array._data

    # Split channels
    size_per_channel = h * w
    R = data[0 * size_per_channel: 1 * size_per_channel]
    G = data[1 * size_per_channel: 2 * size_per_channel]
    B = data[2 * size_per_channel: 3 * size_per_channel]

    # Interleave pixel-wise (R0,G0,B0, R1,G1,B1, ...)
    pixels = [(R[i], G[i], B[i]) for i in range(size_per_channel)]

    # Create and save image
    img = Image.new("RGB", (w, h))
    img.putdata(pixels)
    img.save(path)


image = read_image('./sample.jpg')
# data = image.data
# r = mnp.Array(data[0])
# g = mnp.Array(data[1])
# b = mnp.Array(data[2])


# gr = (0.299 * r + 0.587 * g + 0.114 * b)

# grey = mnp.Array([gr._data for _ in range(image.shape[0])], shape = image.shape, element_type=int)

# red = mnp.Array([[(1 * r)._data, (0.1 * g)._data, (0.1 * b)._data]], shape = image.shape, element_type=int)

# consttrast = mnp.Array([[(0.1 * r)._data, (0.1 * g)._data, (0.1 * b)._data]], shape = image.shape, element_type=int)

gr = mnp.array([[0.299, 0, 0],
                [0, 0.587, 0],
                [0, 0, 0.114]])

re = mnp.array([[1, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0.1]])

const = mnp.array([[1, 0, 0],
                   [0, 0.1, 0],
                   [0, 0, 0.1]])

grey = (image.transpose(1, 2, 0) @ gr).transpose(2, 0, 1)

save_image(grey, './grey.jpg')
# save_image(red, './red.jpg')
# save_image(consttrast, './const.jpg')
