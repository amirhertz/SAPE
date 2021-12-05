from custom_types import *
from models import encoding_models, encoding_controler
from utils import files_utils, train_utils, image_utils
import constants
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import imageio


class Function:

    def __call__(self, x: T) -> T:
        return self.function(x)

    @property
    def name(self):
        return self.function.__name__

    def __init__(self, function: Callable[[T], T]):
        self.function = function
        self.samples = None


def export_poly(vs: Optional[T], dots: Optional[T], path: str, colors: Optional[TNS] = None, r: int = 6,
                opacity: bool = False):

    res = 1000

    def to_rgba(c: T):
        c = (c * 255).long().tolist() + [0]
        c = tuple(c)
        return c

    points = []
    for pts in (vs, dots):
        if pts is not None:
            pts = pts.clone()
            pts = pts.clone()
            pts[:, 0] = (pts[:, 0] + 1) / 2
            pts[:, 1] = 1 - (pts[:, 1] + 1) / 2
            pts = (pts * (res - 200) + 100).long().tolist()
            pts = [tuple(xy) for xy in pts]
        points.append(pts)

    image = Image.new("RGB", (res, res), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    if vs is not None:
        if colors is None or colors[0] is None:
            draw.line(points[0], fill='black', width=2)
        else:
            for i in range(len(points[0]) - 1):
                fill = to_rgba(colors[0][i])
                line = points[0][i: i + 2]
                draw.line(line, fill=fill, width=1)
    if dots is not None:
        for i, (x, y) in enumerate(points[1]):
            if colors is not None and colors[1] is not None:
                fill = to_rgba(colors[1][i])
                outline = (80, 80, 80)
            else:
                fill = (255, 0, 0, 0)
                outline = None
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, width=1, outline=outline)

    # plt.imshow(V(image))
    # plt.show()
    path = files_utils.add_suffix(path, '.png')
    files_utils.init_folders(path)
    # if not constants.DEBUG:
    if opacity:
        image = V(image)
        mask = np.sum(image, axis=2) == 255 * 3
        alpha = np.ones((*image.shape[:-1], 1), dtype=image.dtype) * 255
        alpha[mask] = 0
        image = np.concatenate((image, alpha), axis=2)
        imageio.imsave(path, image)
    else:
        image.save(path)


def init_source_target(func: Function, num_samples: int):
    vs_base = torch.linspace(-1, 1, 3000).unsqueeze(-1)
    target_func = func(vs_base)
    target_func = torch.cat((vs_base, target_func), dim=1)
    if func.samples is None:
        vs_in = torch.linspace(-1, 1, num_samples + 1).unsqueeze(-1)
        vs_in += (vs_in[1, 0] - vs_in[0, 0]) / 2
        vs_in[-1] = .99
        labels = func(vs_in)
    else:
        vs_in = func.samples[:, 0].unsqueeze(-1)
        labels = func.samples[:, 1].unsqueeze(-1)
    return vs_base, vs_in, labels, target_func


def optimize(func: Function, encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams,
             num_samples: int, device: D,
             freq=500, verbose=False):
    name = func.name
    vs_base, vs_in, labels, target_func = init_source_target(func, num_samples)
    vs_base, vs_in, labels, target_func = vs_base.to(device), vs_in.to(device), labels.to(device), target_func.to(
        device)
    lr = 1e-5
    model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
    tag = f'{encoding_type.value}_{controller_type.value}'
    if encoding_type is EncodingType.NoEnc:
        lr = 1e-4
    block_iterations = model.block_iterations
    out_path = f'{constants.CHECKPOINTS_ROOT}/1d/{name}/'
    opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    export_poly(target_func, torch.cat((vs_in, labels), dim=1), f'{out_path}target.png', opacity=True)
    for i in range(control_params.num_iterations):
        opt.zero_grad()
        out = model(vs_in)
        loss_all = nnf.mse_loss(out, labels, reduction='none')
        loss = loss_all.mean()
        loss.backward()
        opt.step()
        model.stash_iteration(loss_all.squeeze(-1))
        if block_iterations > 0 and (i + 1) % block_iterations == 0:
            model.update_progress()
        logger.reset_iter('loss', loss)
        if verbose and ((i + 1) % freq == 0 or i == 0):
            out = model(vs_base)
            aprox_func = torch.cat((vs_base, out), dim=1)
            export_poly(aprox_func, torch.cat((vs_in, labels), dim=1), f'{out_path}opt_{tag}/{i:05d}.png')
            if model.is_progressive:
                _, mask_base = model(vs_in, get_mask=True)
                if mask_base.dim() == 1:
                    mask_base = mask_base.unsqueeze(0).expand(vs_in.shape[0], mask_base.shape[0])
                hm_base = mask_base.sum(1) / mask_base.shape[1]
                hm_base = image_utils.to_heatmap(hm_base)
                export_poly(aprox_func, torch.cat((vs_in, labels), dim=1), f'{out_path}heatmap_{tag}/{i:05d}.png',
                            colors=(None, hm_base))
    logger.stop()
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    if verbose and model.is_progressive:
        image_utils.gifed(f'{out_path}heatmap_{tag}/', .03, tag, reverse=False)
        files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
    if verbose:
        image_utils.gifed(f'{out_path}opt_{tag}/', .03, tag, reverse=False)
        files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])


def psine(x: T):
    a,  c, b = .01, 4.5, 4

    def f_(x_):
        return a * (x_ * 2 + b) ** c

    return .5 * torch.sin(f_(x))


def main() -> int:
    device = CUDA(0)
    encoding_types = (EncodingType.NoEnc, EncodingType.FF, EncodingType.FF)
    controller_types = (ControllerType.NoControl, ControllerType.NoControl, ControllerType.SpatialProgressionStashed)
    func = Function(psine)
    num_samples = 25
    control_params = encoding_controler.ControlParams(num_iterations=20000, epsilon=1e-5, res=num_samples//2)
    model_params = encoding_models.ModelParams(domain_dim=1, output_channels=1, num_freqs=256,
                                               hidden_dim=32, std=5., num_layers=2)

    for encoding_type, controller_type in zip(encoding_types, controller_types):
        optimize(func, encoding_type, model_params, controller_type, control_params, num_samples, device, freq=25, verbose=True)
    return 0


if __name__ == '__main__':
    exit(main())
