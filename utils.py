# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Just need the camera functions from
# https://github.com/facebookresearch/pytorch3d/blob/34f648ede0e52a38dfc378ad38b191ee7f1d2855/pytorch3d/renderer/cameras.py

import io
import sys
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence, Tuple, Union
import logging

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

Device = Union[str, torch.device]

class Video():
    def __init__(self, path, name='video_log.mp4', mode='I', fps=30, codec='libx264', bitrate='16M') -> None:
        
        if path[-1] != "/":
            path += "/"
            
        self.writer = imageio.get_writer(path+name, mode=mode, fps=fps, codec=codec, bitrate=bitrate)
    
    def ready_image(self, image, write_video=True):
        # assuming channels last - as renderer returns it
        if len(image.shape) == 4: 
            image = image.squeeze(0)[..., :3].detach().cpu().numpy()
        else:
            image = image[..., :3].detach().cpu().numpy()

        image = np.clip(np.rint(image*255.0), 0, 255).astype(np.uint8)

        if write_video:
            self.writer.append_data(image)

        return image

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")
    
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def visualize_so3_probabilities(
    rotations,
    probabilities,
    rotations_gt=None,
    ax=None,
    fig=None,
    display_threshold_probability=0,
    to_image=True,
    show_color_wheel=True,
    canonical_rotation=np.eye(3),
):
    """Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
        rotations: [N, 3, 3] tensor of rotation matrices
        probabilities: [N] tensor of probabilities
        rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
        ax: The matplotlib.pyplot.axis object to paint
        fig: The matplotlib.pyplot.figure object to paint
        display_threshold_probability: The probability threshold below which to omit
        the marker
        to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
        show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
        canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.
    Returns:
        A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        eulers = matrix_to_euler_angles(torch.tensor(rotation), "XYZ")
        eulers = eulers.numpy()
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(
            longitude,
            latitude,
            s=2500,
            edgecolors=color if edgecolors else "none",
            facecolors=facecolors if facecolors else "none",
            marker=marker,
            linewidth=4,
        )

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=112)
        ax = fig.add_subplot(111, projection="mollweide")
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[None]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 4e3
    eulers_queries = matrix_to_euler_angles(
        torch.tensor(display_rotations), "XYZ"
    )
    eulers_queries = eulers_queries.numpy()
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])

    which_to_display = probabilities > display_threshold_probability

    if rotations_gt is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, "o")
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(
                ax, rotation, "o", edgecolors=False, facecolors="#ffffff"
            )

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2.0 / np.pi),
    )

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.0
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap, shading="auto")
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
        ax.set_xticklabels(
            [r"90$\degree$", r"180$\degree$", r"270$\degree$", r"0$\degree$",],
            fontsize=14,
        )
        ax.spines["polar"].set_visible(False)
        plt.text(
            0.5,
            0.5,
            "Tilt",
            fontsize=14,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    if to_image:
        return plot_to_image(fig)
    else:
        return fig


def plot_to_image(figure):
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format="raw", dpi=112)
    plt.close(figure)
    buffer.seek(0)
    image = np.reshape(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8),
        newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1),
    )
    return image[..., :3]

def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
    
def generate_random_rotations(n=1, device="cpu"):
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    return quaternion_to_matrix(quats)

def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.
    Args:
        device: Device (as str or torch.device)
    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device

def format_tensor(
    input,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.
    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: Device (as str or torch.device) on which the tensor should be placed.
    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    device_ = make_device(device)
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device_)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device_:
        return input

    input = input.to(device=device)
    return input

def convert_to_tensors_and_broadcast(
    *args,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.
    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.
    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    return args_Nd

def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.
    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.
    The vectors are broadcast against each other so they all have shape (N, 1).
    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    broadcasted_args = convert_to_tensors_and_broadcast(
        distance, elevation, azimuth, device=device
    )
    dist, elev, azim = broadcasted_args
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)

def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: Device = "cpu"
) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.
    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)
    The vectors are broadcast against each other so they all have shape (N, 3).
    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)

def look_at_view_transform(
    dist: float = 1.0,
    elev: float = 0.0,
    azim: float = 0.0,
    degrees: bool = True,
    eye: Optional[Union[Sequence, torch.Tensor]] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device: Device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].
    Args:
        dist: distance of the camera from the object
        elev: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
        dist, elev and azim can be of shape (1), (N).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will override the camera position derived from dist, elev, azim.
        up: the direction of the x axis in the world coordinate system.
        at: the position of the object(s) in world coordinates.
        eye, up and at can be of shape (1, 3) or (N, 3).
    Returns:
        2-element tuple containing
        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.
    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        eye, at, up = broadcasted_args
        C = eye
    else:
        broadcasted_args = convert_to_tensors_and_broadcast(
            dist, elev, azim, at, up, device=device
        )
        dist, elev, azim, at, up = broadcasted_args
        C = (
            camera_position_from_spherical_angles(
                dist, elev, azim, degrees=degrees, device=device
            )
            + at
        )

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T

def look_at_view_transform(
    dist=1.0,
    elev=0.0,
    azim=0.0,
    degrees: bool = True,
    eye: Optional[Union[Sequence, torch.Tensor]] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device: Device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].

    Args:
        dist: distance of the camera from the object
        elev: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
        dist, elev and azim can be of shape (1), (N).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will override the camera position derived from dist, elev, azim.
        up: the direction of the x axis in the world coordinate system.
        at: the position of the object(s) in world coordinates.
        eye, up and at can be of shape (1, 3) or (N, 3).

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        eye, at, up = broadcasted_args
        C = eye
    else:
        broadcasted_args = convert_to_tensors_and_broadcast(
            dist, elev, azim, at, up, device=device
        )
        dist, elev, azim, at, up = broadcasted_args
        C = (
            camera_position_from_spherical_angles(
                dist, elev, azim, degrees=degrees, device=device
            )
            + at
        )

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T


def _convert_image_to_rgb(image):
    return image.convert("RGB")




def generate_equivolumetric_grid(recursion_level=3):
    """
    Generates an equivolumetric grid on SO(3).

    Uses a Healpix grid on S2 and then tiles 6 * 2 ** recursion level over 2pi.

    Code adapted from https://github.com/google-research/google-research/blob/master/
        implicit_pdf/models.py

    Grid sizes:
        1: 576
        2: 4608
        3: 36864
        4: 294912
        5: 2359296
        n: 72 * 8 ** n

    Args:
        recursion_level: The recursion level of the Healpix grid.

    Returns:
        tensor: rotation matrices (N, 3, 3).
    """
    import healpy

    log = logging.getLogger("healpy")
    log.setLevel(logging.ERROR)  # Supress healpy linking warnings.

    number_per_side = 2 ** recursion_level
    number_pix = healpy.nside2npix(number_per_side)
    s2_points = healpy.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = torch.tensor(np.stack([*s2_points], 1))

    azimuths = torch.atan2(s2_points[:, 1], s2_points[:, 0])
    # torch doesn't have endpoint=False for linspace yet.
    tilts = torch.tensor(
        np.linspace(0, 2 * np.pi, 6 * 2 ** recursion_level, endpoint=False)
    )
    polars = torch.arccos(s2_points[:, 2])
    grid_rots_mats = []
    for tilt in tilts:
        rot_mats = euler_angles_to_matrix(
            torch.stack(
                [azimuths, torch.zeros(number_pix), torch.zeros(number_pix)], 1
            ),
            "XYZ",
        )
        rot_mats = rot_mats @ euler_angles_to_matrix(
            torch.stack([torch.zeros(number_pix), torch.zeros(number_pix), polars], 1),
            "XYZ",
        )
        rot_mats = rot_mats @ euler_angles_to_matrix(
            torch.tensor([[tilt, 0.0, 0.0]]), "XYZ"
        )
        grid_rots_mats.append(rot_mats)

    grid_rots_mats = torch.cat(grid_rots_mats, 0)
    return grid_rots_mats.float()



    
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])
