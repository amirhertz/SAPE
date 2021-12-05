from utils.image_utils import init_source_target
from custom_types import *
from models import encoding_controler, encoding_models
from utils import files_utils, train_utils, image_utils
import constants


def plot_image(model: encoding_controler.EncodedController, vs_in: T, ref_image: ARRAY):
    model.eval()
    with torch.no_grad():
        if model.is_progressive:
            out, mask = model(vs_in, get_mask=True)
            if mask.dim() != out.dim():
                mask: T = mask.unsqueeze(0).expand(out.shape[0], mask.shape[0])
            hm = mask.sum(1) / mask.shape[1]
            hm = image_utils.to_heatmap(hm)
            hm = hm.view(*ref_image.shape[:-1], 3)
        else:
            out = model(vs_in, get_mask=True)
            hm = None
        out = out.view(ref_image.shape)
    model.train()
    return out, hm


def optimize(image_path: Union[ARRAY, str], encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams, group, device: D,
             freq: int, verbose=False):

    def shuffle_coords():
        nonlocal vs_in, labels
        order = torch.rand(vs_in.shape[0]).argsort()
        vs_in, labels = vs_in[order], labels[order]

    name = files_utils.split_path(image_path)[1]
    vs_base, vs_in, labels, target_image, image_labels, masked_image = group
    tag = f'{name}_{encoding_type.value}_{controller_type.value}'
    out_path = f'{constants.CHECKPOINTS_ROOT}/2d_images/{name}/'
    lr = 1e-3
    model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
    block_iterations = model.block_iterations
    vs_base, vs_in, labels, image_labels = vs_base.to(device), vs_in.to(device), labels.to(device), image_labels.to(device)
    opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    files_utils.export_image(target_image, f'{out_path}target.png')
    if masked_image is not None:
        files_utils.export_image(masked_image, f'{out_path}target_masked.png')
    for i in range(control_params.num_iterations):
        opt.zero_grad()
        out = model(vs_in)
        loss_all = nnf.mse_loss(out, labels, reduction='none')
        loss = loss_all.mean()
        loss.backward()
        opt.step()
        model.stash_iteration(loss_all.mean(-1))
        logger.stash_iter('mse_train', loss)
        shuffle_coords()
        if block_iterations > 0 and (i + 1) % block_iterations == 0:
            model.update_progress()
        if (i + 1) % freq == 0 and verbose:
            with torch.no_grad():
                out, hm = plot_image(model, vs_base, target_image)
                if hm is not None:
                    files_utils.export_image(hm, f'{out_path}heatmap_{tag}/{i:04d}.png')
                files_utils.export_image(out, f'{out_path}opt_{tag}/{i:04d}.png')
        logger.reset_iter()
    logger.stop()
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    if verbose:
        image_utils.gifed(f'{out_path}opt_{tag}/', .07, tag, reverse=False)
        if model.is_progressive:
            image_utils.gifed(f'{out_path}heatmap_{tag}/', .07, tag, reverse=False)
            files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
                                   filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
        files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])


def main() -> int:
    device = CUDA(0)
    image_path = files_utils.get_source_path()
    name = files_utils.split_path(image_path)[1]
    scale = .25
    group = init_source_target(image_path, name, scale=scale, max_res=512, square=False)
    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_freqs=256,
                                               hidden_dim=256, std=20., num_layers=3)
    control_params = encoding_controler.ControlParams(num_iterations=5000, epsilon=1e-3, res=128)
    encoding_types = (EncodingType.NoEnc, EncodingType.FF, EncodingType.FF)
    controller_types = (ControllerType.NoControl, ControllerType.NoControl, ControllerType.SpatialProgressionStashed)
    for encoding_type, controller_type in zip(encoding_types, controller_types):
        optimize(image_path, encoding_type, model_params, controller_type, control_params, group, device,
                 50, verbose=True)
    return 0


if __name__ == '__main__':
    exit(main())
