from custom_types import *
import constants
from models import encoding_models, encoding_controler
from utils import files_utils, train_utils, mesh_utils, image_utils
import cv2 as cv
from PIL import  Image, ImageDraw
import imageio


def calibrate(model: encoding_controler.EncodedController, vs_in: T, target, epsilon=constants.EPSILON, max_iters=5000):
    optimizer = Optimizer(model.parameters(), lr=1e-4)
    logger = train_utils.Logger()
    logger.start(max_iters)
    for i in range(max_iters):
        optimizer.zero_grad()
        out = model(vs_in)
        loss = nnf.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        logger.stash_iter('mse', loss)
        logger.reset_iter()
        if loss.le(epsilon).item():
            break
    logger.stop(False)


def chamfer(vs_base: T, vs_target: T, dist, num_samples=1000):
    device = vs_base.device
    target_select = torch.rand(vs_target.shape[0], device=device).argsort()[: min(vs_target.shape[0], num_samples)]
    target_samples = vs_target[target_select]
    base_samples = vs_base
    dist_a, dist_b, close_to_a, close_to_b = mesh_utils.simple_chamfer(base_samples, target_samples, aggregate=False)
    return dist_a.mean() + dist_b.mean(), dist_a


class Silhouette:

    @staticmethod
    def export_poly(vs: T, path:str, fill: Tuple[int, int, int] = (180, 180, 180),
                    bg: Tuple[int, int, int] = (255, 255, 255), heatmap: TN = None):

        def to_rgba(c: T):
            c = (c * 255).long().tolist() + [0]
            c = tuple(c)
            return c
        res = 512
        vs = (vs + Silhouette.GLOBAL_SCALE) * ((res - 40) / (2 * Silhouette.GLOBAL_SCALE)) + 20
        points = vs.long().tolist()
        points.append(points[0])
        points = [tuple(xy) for xy in points]
        image = Image.new("RGB", (res, res), color=bg)
        draw = ImageDraw.Draw(image)
        draw.polygon((tuple(points)), fill=fill)
        if heatmap is not None:
             # else:
            for i in range(len(points) - 1):
                line_color = to_rgba(heatmap[i])
                line = points[i: i + 2]
                draw.line(line, fill=line_color, width=7)
        else:
            draw.line(points, fill='black', width=5)
        path = files_utils.add_suffix(path, '.png')
        files_utils.init_folders(path)
        # image = image.resize((512, 512), resample=Image.ANTIALIAS)
        image = V(image)
        mask = np.sum(image, axis=2) == 255 * 3
        alpha = np.ones((*image.shape[:-1], 1), dtype=image.dtype) * 255
        alpha[mask] = 0
        image = np.concatenate((image, alpha), axis=2)
        imageio.imsave(path, image)

    @staticmethod
    def inside_is_white(image: ARRAY) -> bool:
        borders = image[0, :], image[-1, :], image[:, 0], image[:, -1]
        borders = np.concatenate(borders, axis=0)
        is_white = np.less_equal(borders, 100).all()
        return is_white

    @staticmethod
    def image2points(path: str):
        extension = files_utils.split_path(path)[-1]
        if extension == '.obj':
            vs = files_utils.load_mesh(path)[:, :2]
            vs[:, 1] = -vs[:, 1]
        else:
            image = files_utils.load_image(path)
            if image.ndim == 3:
                image = image.mean(-1)
            silhouette = np.less_equal(image, 240)
            if Silhouette.inside_is_white(image):
                silhouette = ~silhouette
            contours, hierarchy = cv.findContours(silhouette.astype(np.uint8), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
            contours = torch.from_numpy(contours[0][:, 0]).float()
            contours_ = torch.cat((contours, contours[0].unsqueeze(0)), dim=0)
            lengths = (contours_[1:] - contours_[:-1]).norm(2, 1)
            min_length = lengths.min() / 2
            max_length = lengths.max()
            while max_length > min_length:
                contours_ = torch.cat((contours, contours[0].unsqueeze(0)), dim=0)
                lengths = (contours_[1:] - contours_[:-1]).norm(2, 1)
                mask = lengths.gt(min_length)
                max_length = lengths.max()
                mids = (contours_[1:] + contours_[:-1]) / 2
                mask = torch.stack((torch.ones(contours.shape[0], dtype=torch.bool), mask), dim=1).flatten()
                contours = torch.stack((contours, mids), dim=1).view(-1, 2)[mask]
            vs = contours
        vs = mesh_utils.to_unit_sphere(vs, scale=Silhouette.GLOBAL_SCALE)
        return vs

    @staticmethod
    def get_circle(num_points: int):
        samples = torch.linspace(0, 2, num_points + 2)[1:-1].unsqueeze(1)
        y = torch.sin(samples * np.pi)
        x = torch.cos(samples * np.pi)
        vs = torch.cat((x, y), dim=1)
        return samples - 1, vs * Silhouette.GLOBAL_SCALE

    GLOBAL_SCALE = 1

    def __init__(self, filename: str, num_points: int, device: D):
        self.name = files_utils.split_path(filename)[1]
        self.target_pts = self.image2points(filename).to(device)
        self.source_points, self.source_calibrate = self.get_circle(num_points)
        self.source_points, self.source_calibrate = self.source_points.to(device), self.source_calibrate.to(device)


def optimize(silhouette: Silhouette, encoding_type: EncodingType, model_params: encoding_models.ModelParams,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams, device: D, freq: int, verbose=True):
    name = silhouette.name
    dist = torch.distributions.categorical.Categorical(probs=torch.ones(silhouette.source_points.shape[0], device=device))
    tag = f'{encoding_type.value}_{controller_type.value}'
    lr = 1e-5
    if encoding_type is EncodingType.NoEnc:
        lr = 1e-4
    model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
    out_path = f'{constants.CHECKPOINTS_ROOT}/silhouette/{name}/'
    silhouette.export_poly(silhouette.target_pts, f'{out_path}target')
    opt = Optimizer(model.parameters(), lr=lr)
    calibrate(model, silhouette.source_points, silhouette.source_calibrate)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    for i in range(control_params.num_iterations):
        opt.zero_grad()
        out = model(silhouette.source_points)
        loss, loss_all = chamfer(out, silhouette.target_pts, dist)
        loss.backward()
        opt.step()
        model.stash_iteration(loss_all)
        logger.stash_iter('loss', loss)
        if verbose and ((i + 1) % freq == 0 or i == 0):
            with torch.no_grad():
                silhouette.export_poly(out, f'{out_path}opt_{tag}/{i:05d}')
        if model.block_iterations > 0 and (i + 1) % model.block_iterations == 0:
            model.update_progress()
        logger.reset_iter()
    logger.stop()
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    if verbose:
        image_utils.gifed(f'{out_path}opt_{tag}/', .07, tag, reverse=False)
        files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1:05d}' == x[1])


def main() -> int:
    device = CUDA(0)
    encoding_types = (EncodingType.NoEnc, EncodingType.FF, EncodingType.FF)
    controller_types = (ControllerType.NoControl,ControllerType.NoControl, ControllerType.SpatialProgressionStashed)
    path = files_utils.get_source_path()
    control_params = encoding_controler.ControlParams(num_iterations=10000, epsilon=1e-3, res=128, num_blocks=6)
    model_params = encoding_models.ModelParams(domain_dim=1, output_channels=2, std=5., num_layers=2)
    silhouette = Silhouette(path, 500, device)
    for encoding_type, controller_type in zip(encoding_types, controller_types):
        optimize(silhouette, encoding_type, model_params, controller_type, control_params, device, 25, verbose=True)
    return 0


if __name__ == '__main__':
    exit(main())
