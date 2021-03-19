import rasterio


def read_tif(path, *args, **kwargs):
    """
    Args:
        path (string): Path to the tif file.

    Returns:
        numpy.ndarray: Numpy array of shape (CxHxW)
    """
    with rasterio.open(path) as source:
        bands = source.read()
        return bands
