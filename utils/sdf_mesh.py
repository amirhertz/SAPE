'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

import plyfile
import skimage.measure
import time
from custom_types import *
from utils import files_utils
from utils.train_utils import Logger


def create_mesh(decoder: Union[nn.Module, Callable[[T], T]], filename, res=256, max_batch=32 ** 3, offset=None, scale=1, device=CPU, verbose=False):
    start = time.time()
    ply_filename = filename

    # decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (res - 1)

    overall_index = torch.arange(0, res ** 3, 1, dtype=torch.int64)
    samples = torch.zeros(res ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % res
    samples[:, 1] = torch.div(overall_index.long(), res, rounding_mode='floor') % res
    samples[:, 0] = torch.div(torch.div(overall_index.long(), res, rounding_mode='floor'), res, rounding_mode='floor') % res

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = res ** 3

    samples.requires_grad = False
    num_iters = num_samples // max_batch + int(num_samples % max_batch != 0)
    if verbose:
        logger = Logger()
        logger.start(num_iters, tag='meshing')
    with torch.no_grad():
        for i in range(num_iters):
            sample_subset = samples[i * max_batch: min((i + 1) * max_batch, num_samples), 0:3].to(device)
            samples[i * max_batch: min((i + 1) * max_batch, num_samples), 3] = (
                decoder(sample_subset * scale)
                .squeeze()
                .detach()
                .cpu()
            )
            if verbose:
                logger.reset_iter()
    if verbose:
        logger.stop()
    sdf_values = samples[:, 3]
    # return sdf_values, samples[:, :3]
    sdf_values = sdf_values.reshape(res, res, res)

    end = time.time()
    # print("sampling took: %f" % (end - start))

    return convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename,
        offset,
        None,
        device=device
    )


def convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor, voxel_grid_origin, voxel_size,
                               ply_filename_out, offset=None, scale=None, device: D = CPU):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except BaseException:
        print("sdf fail")
        return None

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    if ply_filename_out:
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        ply_filename_out = files_utils.add_suffix(ply_filename_out, '.ply')
        files_utils.init_folders(ply_filename_out)
        ply_data.write(ply_filename_out)

    return torch.from_numpy(mesh_points.copy()).float().to(device), torch.from_numpy(faces.copy()).long().to(device)


