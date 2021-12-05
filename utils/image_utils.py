import constants
import scipy.ndimage
from matplotlib import pyplot as plt
from PIL import Image
from custom_types import *
from custom_types import T, ARRAY
from utils import files_utils
import imageio


def is_grayscale(image: ARRAY) -> bool:
    if image.ndim == 2 or image.shape[-1] == 1:
        return True
    mask = np.equal(image[:, :, 0], image[:, :, 1]) * np.equal(image[:, :, 2], image[:, :, 1])
    return mask.all()


def crop_square(image:ARRAY) -> ARRAY:
    h, w, c = image.shape
    offset = abs(h - w)
    if h > w:
        image = image[offset // 2:  h - offset + offset // 2]
    elif h < w:
        image = image[:, offset // 2:  w - offset + offset // 2]
    return image


def resize(image_arr: ARRAY, max_edge_length: int) -> ARRAY:
    h, w, c = image_arr.shape
    max_edge = max(w, h)
    if max_edge < max_edge_length:
        return image_arr
    if c == 1:
        image_arr = image_arr[:, :, 0]
    image = Image.fromarray(image_arr)
    s = max_edge_length / float(max_edge)
    size = (int(w * s), int(h * s))
    image = V(image.resize(size, resample=Image.BICUBIC))
    if c == 1:
        image = np.expand_dims(image, 2)
    return image


def rba_to_rgb(path: str):

    rgba_image = Image.open(path)
    rgba_image.load()
    background = Image.new("RGB", rgba_image.size, (255, 255, 255))
    background.paste(rgba_image, mask = rgba_image.split()[3])
    return V(background)


def gifed(folder: str, interval: float, name: str, filter_by: Optional[Callable[[List[str]], bool]] = None,
          loop: int = 0, split: int = 1, reverse: bool = True, mp4=False, is_alpha: bool = False):
    folder = files_utils.add_suffix(folder, "/")
    files = files_utils.collect(folder, '.png')
    if filter_by is not None:
        files = list(filter(filter_by, files))
    files = sorted(files, key=lambda x: x[1])
    # files = sorted(files, key=lambda x: int(x[1].split('_L')[-1]))
    if len(files) > 0:
        if is_alpha:
            images = [[rba_to_rgb(''.join(file)) for file in files]]
        else:
            images = [[imageio.imread(''.join(file)) for file in files]]
        # images = [[np.transpose(image, (1,0,2)) for image in images[0]]]
        if split > 1:
            images_ = []
            for i, image in enumerate(images[0]):
                if i % split == 0:
                    images_.append([])
                images_[-1].append(image)
            images = images_
        for i, group in enumerate(images):
            if reverse:
                group_ = group.copy()
                group_.reverse()
                group = group + group_
                interval_ = interval
            else:
                interval_ = [interval] * len(group)
                interval_[0] = 1
                # interval_[-1] = 1.5
            extension = 'mp4' if mp4 else 'gif'
            if mp4:
                fps = (1. / interval)
                imageio.mimsave(f'{folder}{name}{str(i) if split > 1 else ""}.{extension}', group, fps=fps)
            else:
                imageio.mimsave(f'{folder}{name}{str(i) if split > 1 else ""}.{extension}',
                                group, duration=interval_, loop=loop)



def to_heatmap(vals: Union[T, ARRAY], palette: str = 'coolwarm') -> T:
    shape = vals.shape
    if type(vals) is T:
        vals: ARRAY = vals.detach().cpu().numpy()
    to_reshape = vals.ndim > 1
    if to_reshape:
        vals = vals.flatten()
    vals = (vals * 255).astype(np.uint8)
    colormap = plt.get_cmap(palette)
    np_heatmap = colormap(vals)[:, :3]
    # np_heatmap = np.ascontiguousarray(cv2.applyColorMap(np_vals, cv2.COLORMAP_HOT)[:, 0, ::-1])
    heatmap = torch.from_numpy(np_heatmap).float()
    if to_reshape:
        heatmap = heatmap.view(*shape, 3)
    return heatmap


def unroll_domain(h: int, w: int) -> T:
    vs_y = torch.linspace(-1, 1., h)
    vs_x = torch.linspace(-1, 1., w)
    vs = torch.meshgrid(vs_y, vs_x)
    vs = torch.stack(vs, dim=2)
    return vs


def random_sampling(image: ARRAY, scale: Union[float, int], non_uniform_sampling=False):
    h, w, c = image.shape
    coords = unroll_domain(h, w).view(-1, 2)
    labels = torch.from_numpy(image).reshape(-1, c).float() / 255
    masked_image = labels.clone()

    if scale < 1:
        split = int(h * w * scale)
    else:
        split = (h * w) // int(scale ** 2)

    if non_uniform_sampling:
        grayscale = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        edge_map = np.abs(scipy.ndimage.filters.laplace(grayscale))
        weight_map = edge_map / edge_map.sum()

        select = np.random.choice(a=h * w, size=split, replace=False, p=weight_map.reshape(-1))
        select = torch.from_numpy(select)
        masked = torch.ones(h*w)
        masked[select] = 0
        masked = torch.nonzero(masked).squeeze(-1)
    else:
        select = torch.rand(h * w).argsort()
        masked = select[split:]
        select = select[:split]

    sample_cords = coords[select]
    labels = labels[select]
    masked_image[masked] = 1
    masked_image = masked_image.view(h, w, c)
    return labels, sample_cords, coords, masked_image


def grid_sampling(image: ARRAY, scale: int):
    h, w, c = image.shape
    coords = unroll_domain(h, w)
    labels = torch.from_numpy(image)[::scale, ::scale].reshape(-1, c).float() / 255
    # masked_image = labels.clone()
    sample_cords = (coords[::scale, ::scale]).reshape(-1, 2)
    return labels, sample_cords, coords.view(-1, 2), None


def init_source_target(path: Union[ARRAY, str], name: str, max_res: int, scale: Union[float, int],
                       square: bool = True, non_uniform_sampling=False):
    if type(path) is str:
        image = files_utils.load_image(path)
    else:
        image = path
    if is_grayscale(image):
        image = image[:, :, :1]
    if square:
        image = crop_square(image)
    image = resize(image, max_res)
    h, w, c = image.shape
    cache_path = f'{constants.RAW_IMAGES}cache/{name}_{scale}.pkl'
    cache = files_utils.load_pickle(cache_path)
    cache = None
    if cache is None:
        cache = random_sampling(image, scale, non_uniform_sampling=non_uniform_sampling)
        files_utils.save_pickle(cache, cache_path)
    labels, samples, vs_base, masked_image = cache
    image_labels = torch.from_numpy(image).reshape(-1, c).float() / 255
    return vs_base, samples, labels, image, image_labels, masked_image
