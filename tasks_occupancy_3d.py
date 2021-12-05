from custom_types import *
import constants
from utils import mesh_utils, files_utils, train_utils, sdf_mesh, image_utils
from models import encoding_models, encoding_controler
import igl


def get_in_out(mesh: T_Mesh, points: T):
    vs, faces = mesh[0].numpy(), mesh[1].numpy()
    points = points.numpy()
    w = igl.winding_number(vs, faces, points)
    w = torch.from_numpy(w).float()
    labels = w.lt(.9).float()
    return labels


class MeshSampler(Dataset):

    def shuffle(self):
        order = torch.rand(self.points.shape[0]).argsort()
        self.points, self.labels = self.points[order], self.labels[order]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        return self.points[item], self.labels[item]

    def get_random_points(self, num_points: float) -> TS:
        random_points = (torch.rand(int(num_points), 3) * 2 - 1)
        labels = get_in_out(self.mesh, random_points).unsqueeze(1)
        return random_points, labels

    def get_surface_points(self, num_points: float, sigma: float, on_surface_points: Optional[T] = None) -> TS:
        if on_surface_points is None:
            on_surface_points = mesh_utils.sample_on_mesh(self.mesh, int(num_points), sample_s=mesh_utils.SampleBy.HYB)[0]
        near_points = on_surface_points + torch.randn(on_surface_points.shape) * sigma
        near_points = near_points.clamp_(-1, 1)
        labels = get_in_out(self.mesh, near_points).unsqueeze(1)
        return near_points, labels, on_surface_points

    def init_samples(self, total=1e6) -> TS:
        split = [float(part) / sum(self.split) for part in self.split]
        near_points_a, labels_a, on_surface_points = self.get_surface_points(int(total * split[0]), .01)
        near_points_b, labels_b, _ = self.get_surface_points(int(total * split[0]), .1, on_surface_points)
        random_points, labels_c, = self.get_random_points(int(total - near_points_a.shape[0] * 2))
        all_points = torch.cat((near_points_a, near_points_b, random_points), dim=0)
        labels = torch.cat((labels_a, labels_b, labels_c), dim=0)
        return all_points.cpu(), labels.cpu()

    def reset(self):
        if self.pointer >= len(self.data):
            self.data.append(self.init_samples())
            self.save_data()
        self.points, self.labels = self.data[self.pointer]
        self.points, self.labels = self.points.to(self.device), self.labels.to(self.device)
        self.shuffle()
        self.pointer = (self.pointer + 1) % self.buffer_size

    @staticmethod
    def load_data(mesh_path: str):
        name = files_utils.split_path(mesh_path)[1]
        cache_path = f"{constants.DATA_ROOT}/cache/sdf_{name}"
        data = files_utils.load_pickle(cache_path)
        return data

    def save_data(self):
        if self.buffer_size == len(self.data) and not self.cache_saved:
            name = files_utils.split_path(self.mesh_path)[1]
            files_utils.save_pickle(self.data, f"{constants.DATA_ROOT}/cache/sdf_{name}")
            self.cache_saved = True

    def load_mesh(self):
        mesh = files_utils.load_mesh(self.mesh_path)
        mesh = mesh_utils.to_unit_sphere(mesh, scale=.95)
        mesh = mesh_utils.triangulate_mesh(mesh)[0]
        return mesh

    def __init__(self, path, device: D, buffer_size=10):
        self.device = device
        self.name = files_utils.split_path(path)[1]
        self.data = None #self.load_data(path)
        self.cache_saved = False
        self.split = (3, 3, 3)
        self.buffer_size = buffer_size
        self.mesh_path = path
        self.pointer = 0
        self.mesh = self.load_mesh()
        if self.data is None:
            self.data = []
        else:
            self.cache_saved = len(self.data) >= self.buffer_size
        self.points: TN = None
        self.labels: TN = None


def model_for_export(model) -> Callable[[T], T]:

    def call(x: T) -> T:
        out_: T = model(x)
        out_.sigmoid_()
        out_ = out_ - .5
        return out_
    model.eval()

    return call


def optimize(ds: MeshSampler, encoding_type: EncodingType, model_params: encoding_models.ModelParams,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams,
             device: D, freq: int, verbose=False):

    def export_heatmap():
        model.eval()
        res = control_params.res
        mask = model.mask.view(res, res, res, -1).sum(-1).float() / model.mask.shape[-1]
        mask = mask.view(1, 1, res, res, res)
        mask = nnf.interpolate(mask, mode='trilinear', scale_factor=256 // res, align_corners=True).squeeze()
        for i in range(3):
            for j in range(mask.shape[0]):
                mask_ = mask[:, :, j]
                if i == 1:
                    mask_ = mask_.permute(1, 0)
                mask_ = mask_.flip(0)
                hm = image_utils.to_heatmap(mask_)
                files_utils.export_image(hm, f'{out_path}heatmap/{j:04d}.png')
            image_utils.gifed(f'{out_path}heatmap/', .1, f'{tag}_{i:d}', reverse=True)
            files_utils.delete_all(f'{out_path}heatmap/', '.png')
            mask = mask.permute(2, 0, 1)
        return

    batch_size = 5000
    name = ds.name
    tag = f'{encoding_type.value}_{controller_type.value}'
    out_path = f'{constants.CHECKPOINTS_ROOT}/3d_occupancy/{name}/'
    ds.reset()
    in_iters = len(ds) // batch_size
    epochs = control_params.num_iterations
    control_params.num_iterations = in_iters * control_params.num_iterations
    model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device).to(device)
    lr = 1e-4
    opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(epochs, tag=f"{name} {tag}")
    for i in range(epochs):
        loss_train = 0
        for j in range(in_iters):
            vs, labels = ds.points[j * batch_size: (j + 1) * batch_size], ds.labels[j * batch_size: (j + 1) * batch_size]
            opt.zero_grad()
            out = model(vs)
            loss_all = nnf.binary_cross_entropy_with_logits(out, labels, reduction='none')
            loss = loss_all.mean()
            loss.backward()
            opt.step()
            model.stash_iteration(loss_all.mean(-1))
            loss_train += loss.item()
        loss_train = float(loss_train) / in_iters
        logger.reset_iter('mse_train', loss_train)
        model.update_progress()
        ds.reset()
        if (i + 1) % freq == 0 and verbose:
            sdf_mesh.create_mesh(model_for_export(model), f'{out_path}{tag}_meshes/{i:04d}', res=128, device=device)
            model.train()
    logger.stop()
    # model.load_state_dict(torch.load(f'{out_path}model_{tag}.pth', map_location=device))
    sdf_mesh.create_mesh(model_for_export(model), f'{out_path}final_{tag}', res=256, device=device)
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    if model.is_progressive:
        export_heatmap()


def main():
    device = CUDA(0)
    mesh_path = files_utils.get_source_path()
    encoding_types = (EncodingType.NoEnc, EncodingType.FF,  EncodingType.FF)
    controller_types = (ControllerType.NoControl, ControllerType.NoControl, ControllerType.SpatialProgressionStashed)
    std = 20
    ds = MeshSampler(mesh_path, device)
    control_params = encoding_controler.ControlParams(num_iterations=500, epsilon=1e-1, res=64)
    model_params = encoding_models.ModelParams(domain_dim=3, output_channels=1, std=std, hidden_dim=256,
                                               num_layers=4, num_frequencies=256)
    for encoding_type, controller_type in zip(encoding_types, controller_types):
        optimize(ds, encoding_type, model_params, controller_type, control_params, device, 25, verbose=False)


if __name__ == '__main__':
    exit(main())
