from custom_types import *
from custom_types import T, TS
from functools import reduce


def scale_all(*values: T):
    # mean_std = [(val.mean(), val.std()) for val in values]
    # values = [val.clamp(scales[0] - scales[1] * 3, scales[0] + scales[1] * 3) for val,scales in zip(values, mean_std)]
    max_val = max([val.max().item() for val in values])
    min_val = min([val.min().item() for val in values])
    scale = max_val - min_val
    values = [(val - min_val) / scale for val in values]
    if len(values) == 1:
        return values[0]
    return values


def get_faces_normals(mesh: Union[T_Mesh, T]) -> T:
    if type(mesh) is not T:
        vs, faces = mesh
        vs_faces = vs[faces]
    else:
        vs_faces = mesh
    if vs_faces.shape[-1] == 2:
        vs_faces = torch.cat(
            (vs_faces, torch.zeros(*vs_faces.shape[:2], 1, dtype=vs_faces.dtype, device=vs_faces.device)), dim=2)
    face_normals = torch.cross(vs_faces[:, 1, :] - vs_faces[:, 0, :], vs_faces[:, 2, :] - vs_faces[:, 1, :])
    return face_normals


def compute_face_areas(mesh: Union[T_Mesh, T]) -> TS:
    face_normals = get_faces_normals(mesh)
    face_areas = torch.norm(face_normals, p=2, dim=1)
    face_areas_ = face_areas.clone()
    face_areas_[torch.eq(face_areas_, 0)] = 1
    face_normals = face_normals / face_areas_[:, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def to_numpy(*tensors: T) -> ARRAYS:
    params = [param.detach().cpu().numpy() if type(param) is T else param for param in tensors]
    return params


def mesh_center(mesh: T_Mesh):
    return mesh[0].mean(0)


def to_center(vs):
    max_vals = vs.max(0)[0]
    min_vals = vs.min(0)[0]
    center = (max_vals + min_vals) / 2
    vs -= center[None, :]
    return vs


def to_unit_sphere(mesh: T_Mesh_T,  in_place: bool = True, scale=1.) -> T_Mesh_T:
    remove_me = 0
    mesh = (mesh, remove_me) if type(mesh) is T else mesh
    vs, faces = mesh
    if not in_place:
        vs = vs.clone()
    vs = to_center(vs)
    norm = vs.norm(2, dim=1).max()
    vs *= scale * norm ** -1
    return vs if faces is remove_me else (vs, faces)


def to_unit_cube(*meshes: T_Mesh_T, scale=1, in_place: bool = True) -> Tuple[Union[T_Mesh_T, Tuple[T_Mesh_T, ...]], TS]:
    remove_me = 0
    meshes = [(mesh, remove_me) if type(mesh) is T else mesh for mesh in meshes]
    vs, faces = meshes[0]
    max_vals = vs.max(0)[0]
    min_vals = vs.min(0)[0]
    max_range = (max_vals - min_vals).max() / 2
    center = (max_vals + min_vals) / 2
    meshes_ = []
    for vs_, faces_ in meshes:
        if not in_place:
            vs_ = vs_.clone()
        vs_ -= center[None, :]
        vs_ /= max_range
        vs_ *= scale
        meshes_.append(vs_ if faces_ is remove_me else (vs_, faces_))
    if len(meshes_) == 1:
        meshes_ = meshes_[0]
    return meshes_, (center, max_range / scale)


def to(tensors, device: D) -> Union[T_Mesh, TS, T]:
    out = []
    for tensor in tensors:
        if type(tensor) is T:
            out.append(tensor.to(device, ))
        elif type(tensor) is tuple or type(tensors) is List:
            out.append(to(list(tensor), device))
        else:
            out.append(tensor)
    if len(tensors) == 1:
        return out[0]
    else:
        return tuple(out)


def clone(*tensors: Union[T, TS]) -> Union[TS, T_Mesh]:
    out = []
    for t in tensors:
        if type(t) is T:
            out.append(t.clone())
        else:
            out.append(clone(*t))
    return out


def normalize(t: T):
    t = t / t.norm(2, dim=1)[:, None]
    return t


def interpulate_vs(mesh: T_Mesh, faces_inds: T, weights: T) -> T:
    vs = mesh[0][mesh[1][faces_inds]]
    vs = vs * weights[:, :, None]
    return vs.sum(1)


def sample_uvw(shape, device: D):
    u, v = torch.rand(*shape, device=device), torch.rand(*shape, device=device)
    mask = (u + v).gt(1)
    u[mask], v[mask] = -u[mask] + 1, -v[mask] + 1
    w = -u - v + 1
    uvw = torch.stack([u, v, w], dim=len(shape))
    return uvw


def get_sampled_fe(fe: T, mesh: T_Mesh, face_ids: T, uvw: TN) -> T:
    # to_squeeze =
    if fe.dim() == 1:
        fe = fe.unsqueeze(1)
    if uvw is None:
        fe_iner = fe[face_ids]
    else:
        vs_ids = mesh[1][face_ids]
        fe_unrolled = fe[vs_ids]
        fe_iner = torch.einsum('sad,sa->sd', fe_unrolled, uvw)
    # if to_squeeze:
    #     fe_iner = fe_iner.squeeze_(1)
    return fe_iner


def sample_on_faces(mesh: T_Mesh,  num_samples: int) -> TS:
    vs, faces = mesh
    uvw = sample_uvw([faces.shape[0], num_samples], vs.device)
    samples = torch.einsum('fad,fna->fnd', vs[faces], uvw)
    return samples, uvw


class SampleBy(Enum):
    AREAS = 0
    FACES = 1
    HYB = 2


def sample_on_mesh(mesh: T_Mesh, num_samples: int, face_areas: TN = None,
                   sample_s: SampleBy = SampleBy.HYB) -> TNS:
    vs, faces = mesh
    if faces is None:  # sample from pc
        uvw = None
        if vs.shape[0] < num_samples:
            chosen_faces_inds = torch.arange(vs.shape[0])
        else:
            chosen_faces_inds = torch.argsort(torch.rand(vs.shape[0]))[:num_samples]
        samples = vs[chosen_faces_inds]
    else:
        weighted_p = []
        if sample_s == SampleBy.AREAS or sample_s == SampleBy.HYB:
            if face_areas is None:
                face_areas, _ = compute_face_areas(mesh)
            face_areas[torch.isnan(face_areas)] = 0
            weighted_p.append(face_areas / face_areas.sum())
        if sample_s == SampleBy.FACES or sample_s == SampleBy.HYB:
            weighted_p.append(torch.ones(mesh[1].shape[0], device=mesh[0].device))
        chosen_faces_inds = [torch.multinomial(weights, num_samples // len(weighted_p), replacement=True) for weights in weighted_p]
        if sample_s == SampleBy.HYB:
            chosen_faces_inds = torch.cat(chosen_faces_inds, dim=0)
        chosen_faces = faces[chosen_faces_inds]
        uvw = sample_uvw([num_samples], vs.device)
        uvw = uvw[:chosen_faces.shape[0]]
        samples = torch.einsum('sf,sfd->sd', uvw, vs[chosen_faces])
    return samples, chosen_faces_inds, uvw


def get_dist_mat(a: T, b: T, batch_size: int = 1000, sqrt: bool = False) -> T:

    iters = a.shape[0] // batch_size
    dist_list = [((a[i * batch_size: (i + 1) * batch_size, None, :] - b[None, :, :]) ** 2).sum(-1)
                 for i in range(iters + 1)]
    all_dist: T = torch.cat(dist_list, dim=0)
    if sqrt:
        all_dist = all_dist.sqrt_()
    return all_dist


def naive_knn(k: int, dist_mat: T, is_biknn=True):

    _, close_to_b = dist_mat.topk(k, 0, largest=False)
    if is_biknn:
        _, close_to_a = dist_mat.topk(k, 1, largest=False)
        return close_to_a, close_to_b.t()
    return close_to_b.t()


def simple_chamfer(a: T, b: T, normals_a=None, normals_b=None, dist_mat: Optional[T] = None,
                   aggregate: bool = True) -> Union[T, TS]:

    def one_direction(fixed: T, search: T, n_f, n_s, closest_id) -> TS:
        min_dist = (fixed - search[closest_id]).norm(2, 1)
        if aggregate:
            min_dist = min_dist.mean()
        if n_f is not None:
            normals_dist = -torch.einsum('nd,nd->n', n_f, n_s[closest_id])
            if aggregate:
                normals_dist = min_dist.mean()
        else:
            normals_dist = 0
        return min_dist, normals_dist

    if dist_mat is None:
        dist_mat = get_dist_mat(a, b)
    close_to_a, close_to_b = naive_knn(1, dist_mat)
    dist_a, dist_a_n = one_direction(a, b, normals_a, normals_b, close_to_a.flatten())
    dist_b, dist_b_n = one_direction(b, a, normals_b, normals_a, close_to_b.flatten())
    if normals_a is None:
        return dist_a, dist_b, close_to_a, close_to_b
    return dist_a, dist_b, dist_a_n, dist_b_n, close_to_a, close_to_b


def is_quad(mesh: Union[T_Mesh, Tuple[T, List[List[int]]]]) -> bool:
    if type(mesh) is T:
        return False
    if type(mesh[1]) is T:
        return False
    else:
        faces: List[List[int]] = mesh[1]
        for f in faces:
            if len(f) == 4:
                return True
    return False


def triangulate_mesh(mesh: Union[T_Mesh, Tuple[T, List[List[int]]]]) -> Tuple[T_Mesh, Optional[T]]:

    def get_skinny(faces_) -> T:
        vs_faces = vs[faces_]
        areas = compute_face_areas(vs_faces)[0]
        edges = reduce(
            lambda a, b: a + b,
            map(
                lambda i: ((vs_faces[:, i] - vs_faces[:, (i + 1) % 3]) ** 2).sum(1),
                range(3)
            )
        )
        skinny_value = np.sqrt(48) * areas / edges
        return skinny_value

    if not is_quad(mesh):
        return mesh, None

    vs, faces = mesh
    device = vs.device
    faces_keep = torch.tensor([face for face in faces if len(face) == 3], dtype=torch.int64, device=device)
    faces_quads = torch.tensor([face for face in faces if len(face) != 3], dtype=torch.int64, device=device)
    faces_tris_a, faces_tris_b = faces_quads[:, :3], faces_quads[:, torch.tensor([0, 2, 3], dtype=torch.int64)]
    faces_tris_c, faces_tris_d = faces_quads[:, 1:], faces_quads[:, torch.tensor([0, 1, 3], dtype=torch.int64)]
    skinny = [get_skinny(f) for f in (faces_tris_a, faces_tris_b, faces_tris_c, faces_tris_d)]
    skinny_ab, skinny_cd = torch.stack((skinny[0], skinny[1]), 1), torch.stack((skinny[2], skinny[3]), 1)
    to_flip = skinny_ab.min(1)[0].lt(skinny_cd.min(1)[0])
    faces_tris_a[to_flip], faces_tris_b[to_flip] = faces_tris_c[to_flip], faces_tris_d[to_flip]
    faces_tris = torch.cat((faces_tris_a, faces_tris_b, faces_keep), dim=0)
    face_twin = torch.arange(faces_tris_a.shape[0], device=device)
    face_twin = torch.cat((face_twin + faces_tris_a.shape[0], face_twin,
                           -torch.ones(faces_keep.shape[0], device=device, dtype=torch.int64)))
    return (vs, faces_tris), face_twin
