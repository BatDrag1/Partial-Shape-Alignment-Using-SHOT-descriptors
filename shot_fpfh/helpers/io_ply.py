"""
Utility functions to read/write .ply files
"""

import logging
import sys
from typing import Protocol

import numpy as np
import numpy.typing as npt

# defining PLY types
ply_dtypes = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)

# numpy reader format
valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def read_ply(filename) -> np.ndarray:
    """
    Reads ".ply" files.

    Args:
        filename (str): the name of the file to read.

    Returns:
        result (numpy.ndarray): the data stored in the file

    Examples:
        Store data in a file
        >>> points = np.random.rand(5, 3)
        >>> values = np.random.randint(2, size=10)
        >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

        Read the file
        >>> data = read_ply('example.ply')
        >>> values = data['values']
        array([0, 0, 1, 1, 0])

        >>> points = np.vstack((data['x'], data['y'], data['z'])).T
        array([[ 0.466  0.595  0.324]
               [ 0.538  0.407  0.654]
               [ 0.850  0.018  0.988]
               [ 0.395  0.394  0.363]
               [ 0.873  0.996  0.092]])

    """

    with open(filename, "rb") as plyfile:
        # check if the file starts with ply
        if b"ply" not in plyfile.readline():
            raise ValueError("The file does not start with the word ply")

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get the extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # Parse header
        num_points, properties = parse_header(plyfile, ext)

        # Get data
        data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):
    # first line describing element vertex
    lines = ["element vertex %d" % field_list[0].shape[0]]

    # property lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append("property %s %s" % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names):
    """
    Writes in ".ply" files.

    Args:
        filename (str): the name of the file to which the data is saved. A '.ply' extension will be appended to the
            file name if it does not already have one.

        field_list ([list, tuple, np.ndarray]): the fields to be saved in the ply file. Either a numpy array, a list of
            numpy arrays or a tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are
            considered.

        field_names (list[str]): the name of each field. It has to be the same length as field_list.

    Examples:
        >>> points = np.random.rand(10, 3)
        >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

        >>> values = np.random.randint(2, size=10)
        >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

        >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
        >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'values']
        >>> write_ply('example3.ply', [points, colors, values], field_names)
    """

    # Format list input to the right form
    field_list = (
        list(field_list) if (isinstance(field_list, (list, tuple))) else [field_list]
    )
    for i, field in enumerate(field_list):
        if field is None:
            logging.warning("WRITE_PLY ERROR: a field is None")
            return False
        elif field.ndim > 2:
            logging.warning("WRITE_PLY ERROR: a field have more than 2 dimensions")
            return False
        elif field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        logging.warning("wrong field dimensions")
        return False

    # check if field_names and field_list have the same number of columns
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        logging.warning("wrong number of field names")
        return False

    # add the extension if not there
    if not filename.endswith(".ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as plyfile:
        # first magical word and encoding format
        header = ["ply", "format binary_" + sys.byteorder + "_endian 1.0"]

        # points properties description
        header.extend(header_properties(field_list, field_names))

        # end of header
        header.append("end_header")

        # write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use to file
    with open(filename, "ab") as plyfile:
        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

    return True


def describe_element(name, df):
    """
    Takes the columns of the dataframe and builds a ply-like description

    Args:
        name (str)
        df (pandas.DataFrame)

    Returns:
        element (List[str])
    """
    property_formats = {"f": "float", "u": "uchar", "i": "int"}
    element = ["element " + name + " " + str(len(df))]

    if name == "face":
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get the first letter of the data type to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append("property " + f + " " + df.columns.values[i])

    return element


class NormalsComputationCallback(Protocol):
    """
    Type signature for callbacks that can compute normals on a point cloud.
    """

    def __call__(
        self,
        query_points: npt.NDArray[np.float64],
        cloud_points: npt.NDArray[np.float64],
        *,
        k: int | None = None,
        radius: float | None = None,
        pre_computed_normals: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        ...


def get_data(
    data_path: str,
    remove_duplicates: bool = False,
    recompute_normals: bool = True,
    k: int | None = None,
    radius: float | None = None,
    normals_computation_callback: NormalsComputationCallback | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    data = read_ply(data_path)

    points = np.vstack((data["x"], data["y"], data["z"])).T
    if "nx" in data.dtype.fields.keys():
        normals = np.vstack((data["nx"], data["ny"], data["nz"])).T
        if recompute_normals:
            logging.info(
                f"Recomputing normals using function {normals_computation_callback.__name__}."
            )
            normals = normals_computation_callback(
                points, points, k=k, radius=radius, pre_computed_normals=normals
            )
    elif "n_x" in data.dtype.fields.keys():
        normals = np.vstack((data["n_x"], data["n_y"], data["n_z"])).T
        if recompute_normals:
            logging.info(
                f"Recomputing normals using function {normals_computation_callback.__name__}."
            )
            normals = normals_computation_callback(
                points, points, k=k, radius=radius, pre_computed_normals=normals
            )
    else:
        if normals_computation_callback is None:
            raise ValueError(
                "The function used to compute normals needs to be specified as the ply file does not contain normals."
            )
        normals = normals_computation_callback(points, points, k=k, radius=radius)

    if remove_duplicates:
        filtered_indexes = np.unique(
            points.round(decimals=4), axis=0, return_index=True
        )[1]
        return points[filtered_indexes], normals[filtered_indexes]

    return points, normals
