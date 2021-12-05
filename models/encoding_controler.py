from custom_types import *
from models import encoding_models
import abc
import functools


class ControlParams:

    def fill_args(self, **kwargs):
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)

    def __init__(self, **kwargs):
        self.res = 64
        self.num_iterations: int = 1000
        self.num_blocks: Optional[int] = None
        self.block_iterations = 256
        self.epsilon = 1e-3
        self.mask_dim = None
        self.fill_args(**kwargs)


class EncodedController(nn.Module, abc.ABC):

    @property
    def is_progressive(self) -> bool:
        return False

    def update_progress(self, *args, **kwargs):
        return

    def stash_iteration(self, *args, **kwargs):
        return

    def update_mask(self):
        return

    @property
    def encoding_dim(self):
        return self.model.encoding_dim

    @property
    def domain_dim(self):
        return self.model.domain_dim

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplemented

    def forward(self, x: T, **kwargs):
        return self.model(x)

    def __init__(self, params: encoding_models.ModelParams, encoding_type: EncodingType, control_params: ControlParams):
        super(EncodedController, self).__init__()
        self.model = encoding_models.get_model(params, encoding_type)
        self.block_iterations = -1


class NoControl(EncodedController):

    @property
    def name(self) -> str:
        return "no_control"


class ProgressiveEncoderController(EncodedController, abc.ABC):

    @property
    def is_progressive(self) -> bool:
        return True

    def stash_iteration(self, *args, **kwargs):
        self.iteration += 1
        with torch.no_grad():
            self.update_mask()

    @abc.abstractmethod
    def update_mask(self):
        raise NotImplemented

    def forward(self, x: T, **kwargs):
        if 'override_mask' in kwargs and kwargs['override_mask'] is not None:
            override_mask = kwargs['override_mask']
        else:
            override_mask = self.mask
        out = self.model(x, override_mask=override_mask)
        if 'get_mask' in kwargs:
            return out, override_mask
        return out

    def init_mask(self):
        return torch.ones(self.model.encoding_dim)

    def load_mask(self):
        mask = torch.zeros(self.mask_stashed.shape[0], self.encoding_dim)
        arange = torch.arange(self.encoding_dim)
        arange = arange.unsqueeze(0).repeat(self.mask_stashed.shape[0], 1)
        fill_a = arange.lt(torch.floor(self.mask_stashed[:, None]).cpu())
        fill_b = ~fill_a * arange.le(self.mask_stashed[:, None].cpu())
        mask[fill_a] = 1
        mask[fill_b] = (self.mask_stashed[self.mask_stashed.lt(self.encoding_dim)] % 1).cpu()
        self.mask = mask

    def to(self, device: D):
        self.mask = self.mask.to(device)
        return super(ProgressiveEncoderController, self).to(device)

    def load_state_dict(self, state_dict, strict: bool = True):
        super(ProgressiveEncoderController, self).load_state_dict(state_dict, strict)
        with torch.no_grad():
            self.load_mask()
            self.mask = self.mask.to(self.mask_stashed.device)

    def save_mask(self):
        self.mask_stashed = self.mask.sum(-1)
        if len(self.mask_stashed .shape) == 0:
            self.mask_stashed = self.mask_stashed.unsqueeze(0)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self.save_mask()
        return super(ProgressiveEncoderController, self).state_dict()

    def __init__(self, params: encoding_models.ModelParams, encoding_type: EncodingType, control_params: ControlParams):
        super(ProgressiveEncoderController, self).__init__(params, encoding_type, control_params)
        self.mask = self.init_mask().detach()
        if self.mask.dim() > 1:
            mask_stashed = torch.zeros(self.mask.shape[0])
        else:
            mask_stashed = torch.zeros(1)
        self.register_buffer('mask_stashed', mask_stashed.detach())
        self.iteration = 0


class GlobalProgressionController(ProgressiveEncoderController):

    def load_mask(self):
        super(GlobalProgressionController, self).load_mask()
        self.mask: T = self.mask.squeeze_()
      
    @property
    def name(self):
        return 'global_progression'

    def increase_block(self):
        self.mask[self.cur_block:  self.next_block] = 1
        self.cur_block = self.next_block
        self.next_block += self.block_size
        if self.model.encoding_dim - self.next_block < self.block_size:
            self.next_block = self.model.encoding_dim

    def update_mask(self):
        if not self.train() or self.iteration > self.progress_iterations:
            return
        elif self.iteration % self.block_iterations == 0:
            self.increase_block()
        else:
            alpha = min(1., float(2 * (self.iteration % self.block_iterations)) / self.block_iterations)
            self.mask[self.cur_block:  self.next_block] = alpha

    def __init__(self, params: encoding_models.ModelParams, encoding_type: EncodingType, control_params: ControlParams):
        super(GlobalProgressionController, self).__init__(params, encoding_type, control_params)
        if control_params.num_blocks is None:
            self.block_size = self.model.domain_dim
            control_params.num_blocks = (self.encoding_dim - self.block_size) // self.block_size
        else:
            self.block_size = self.encoding_dim // control_params.num_blocks
        self.mask[self.block_size:] = 0
        self.cur_block = self.block_size
        self.next_block = self.block_size * 2
        self.block_iterations = 3 * control_params.num_iterations // (4 * control_params.num_blocks)
        self.progress_iterations = self.block_iterations * control_params.num_blocks


class GlobalProgressionControllerEarly(GlobalProgressionController):

    @property
    def name(self):
        return 'global_progression_early'

    def stash_iteration(self, loss: T):
        self.best_score = min(self.best_score, loss.mean().item())
        if self.best_score < self.epsilon and not self.trigger:
            print(f"progress stopped: {self.cur_block} / {self.encoding_dim}")
            self.trigger = True
        super(GlobalProgressionController, self).stash_iteration(loss)

    def update_mask(self):
        if self.best_score < self.epsilon:
            return
        return super(GlobalProgressionControllerEarly, self).update_mask()

    def __init__(self, params: encoding_models.ModelParams, encoding_type: EncodingType, control_params: ControlParams):
        super(GlobalProgressionControllerEarly, self).__init__(params, encoding_type, control_params)
        self.trigger = False
        self.epsilon = control_params.epsilon
        self.best_score = 10000


class SpatialFixedController(ProgressiveEncoderController):

    @property
    def name(self):
        return 'spatial_progression_fixed'

    def blur_loss(self, loss: T):
        if self.domain_dim == 1:
            # return lo
            loss = loss.view(1, 1, loss.shape[0])
            loss = nnf.pad(loss, (1, 1), 'replicate')
            w = torch.ones(1, 1, 3, device=loss.device) / 3
            out = nnf.conv1d(loss, w, stride=1)
            out = out[0, 0].flatten()
        elif self.domain_dim == 2:
            k = 3
            h = w = int(np.sqrt(loss.shape[0]))
            loss = loss.view(1, 1, h, w)
            loss = nnf.pad(loss, (k // 2, k // 2, k // 2, k // 2), 'replicate')
            w = self.get_blur_mask(k, loss.device)
            out = nnf.conv2d(loss, w, stride=1)
            out = out[0, 0].flatten()
        else:
            out = loss
        return out

    def stash_iteration(self, loss: T):
        in_progress = self.in_progress.sum()
        loss = self.blur_loss(loss)
        self.log_buffer[self.iteration % self.buffer_size] = loss.gt(self.epsilon)
        self.in_progress = self.in_progress * self.log_buffer.sum(0).ne(0)
        if self.verbose and self.in_progress.sum() < in_progress:
            print(f"\n{self.in_progress.sum().item():d}: {self.cur_block} / {self.encoding_dim}")
        super(SpatialFixedController, self).stash_iteration(loss)

    def increase_block(self):
        self.mask[self.in_progress, self.cur_block:  self.next_block] = 1
        self.cur_block = self.next_block
        self.next_block += self.block_size
        if self.model.encoding_dim - self.next_block < self.block_size:
            self.next_block = self.model.encoding_dim

    def update_mask(self):
        if not self.train() or self.iteration > self.progress_iterations or not self.in_progress.any():
            return
        elif self.iteration % self.block_iterations == 0:
            self.increase_block()
        else:
            alpha = min(1., float(2 * (self.iteration % self.block_iterations)) / self.block_iterations)
            self.mask[self.in_progress, self.cur_block:  self.next_block] = alpha

    def init_mask(self):
        return torch.ones(self.num_samples, self.model.encoding_dim)

    def interpolate1d(self, x: T) -> T:
        distances: T = ((x[:, None, :] - self.input_example[None, :, :])**2).abs().squeeze()
        indices_const = distances.argmin(0)
        min_dist, indices = distances.topk(2, dim=1, largest=False)
        alpha = min_dist.roll(1, 1) / min_dist.sum(-1)[:, None]
        base_mask = self.get_base_mask().permute(1, 0)
        mask = base_mask[indices]
        mask = torch.einsum('na,nad->nd', alpha, mask)
        first_mask = x.le(self.input_example.min()).squeeze(1)
        last_mask = x.ge(self.input_example.max()).squeeze(1)
        mask[first_mask] = self.mask[0]
        mask[last_mask] = self.mask[-1]
        mask[indices_const] = base_mask
        return mask

    @functools.lru_cache
    def get_blur_mask(self, k, device: D):
        with torch.no_grad():
            w = torch.ones(1, 1, k, k) / k ** 2
        return w.to(device)

    def get_base_mask(self):
        if self.domain_dim == 1:
            # return self.mask.permute(1, 0)
            mask = self.mask.permute(1, 0).unsqueeze(1)
            mask = nnf.pad(mask, (1, 1), 'replicate')
            w = torch.ones(1, 1, 3, device=mask.device) / 3
            out = nnf.conv1d(mask, w, stride=1).squeeze(1)
        elif self.domain_dim == 2:
            k = 3
            h = w = int(np.sqrt(self.mask.shape[0]))
            mask = self.mask.view(h, w, -1).permute(2, 0, 1).unsqueeze(1)
            mask = nnf.pad(mask, (k // 2, k // 2, k // 2, k // 2), 'replicate')
            w = self.get_blur_mask(k, mask.device)
            # w = torch.ones(1, 1, k, k, device=mask.device) / k ** 2
            out = nnf.conv2d(mask, w, stride=1).squeeze(1)
        else:
            return self.mask
        return out

    def interpolate2d(self, x: T) -> T:
        base_mask = self.get_base_mask()
        _, h, w = base_mask.shape
        while h * w != x.shape[0]:
            base_mask = base_mask.unsqueeze(1)
            weight_a = torch.ones(1, 1, 2, 2) / 2
            weight_a[:, :, :, 1] = 0
            mask_pad = nnf.pad(base_mask, (0, 1, 0, 1), mode='replicate')
            mask_a = nnf.conv2d(mask_pad, weight_a.to(base_mask.device), stride=1)
            weight_b = torch.ones(1, 1, 2, 2) / 2
            weight_b[:, :, 1, :] = 0
            mask_b = nnf.conv2d(mask_pad, weight_b.to(base_mask.device), stride=1)
            weight_c = torch.ones(1, 1, 2, 2) / 4
            mask_c = nnf.conv2d(mask_pad, weight_c.to(base_mask.device), stride=1)
            base_mask, mask_a, mask_b, mask_c = [m.squeeze(1).permute(1, 2, 0) for m in (base_mask, mask_a, mask_b, mask_c)]
            mask_a = torch.cat((base_mask, mask_a), dim=1).view(h * 2, w, -1)
            mask_b = torch.cat((mask_b, mask_c), dim=1).view(h * 2, w, -1)
            mask = torch.cat((mask_a, mask_b), dim=2).view(h * 2, w * 2, -1)
            base_mask = mask.permute(2, 0, 1)
            _, h, w = base_mask.shape
        base_mask = base_mask.permute(1, 2, 0)
        base_mask = base_mask.view(-1, base_mask.shape[2])
        return base_mask

    def flat_inds(self, indices: List[TS], alphas: List[TS], res) -> TS:
        format_string = lambda x: ("{" f"0:0{self.domain_dim}b" + "}").format(x)
        mask_inds, mask_alphas = [], []
        for i in range(2 ** self.domain_dim):
            select = format_string(i)
            cur_ind, cur_alpha = 0, 1
            for j, (s, inds, alpha) in enumerate(zip(select, indices, alphas)):
                s = int(s)
                cur_ind += inds[s].clip(0, res - 1) * res ** j
                cur_alpha *= alpha[s]
            mask_inds.append(cur_ind.long())
            mask_alphas.append(cur_alpha)
        mask_inds = torch.stack(mask_inds, 1)
        mask_alphas = torch.stack(mask_alphas, 1)
        return mask_inds, mask_alphas

    def interpolate(self, x) -> T:
        if self.domain_dim == 1:
            return self.interpolate1d(x)
        elif self.domain_dim == 2:
            return self.interpolate2d(x)
        else:
            raise NotImplemented

    def forward(self, x: T, **kwargs):
        if 'override_mask' in kwargs:
            return self.model(x, **kwargs)
        with torch.no_grad():
            if x.shape[0] == self.num_samples:
                mask = self.get_base_mask()
                if self.domain_dim == 1:
                    mask = mask.permute(1, 0)
                if self.domain_dim == 2:
                    mask = mask.permute(1, 2, 0)
                mask = mask.view(x.shape[0], -1)
            else:
                mask = self.interpolate(x)
        out = self.model(x, override_mask=mask)
        if 'get_mask' in kwargs:
            return out, mask
        return out

    def __init__(self, model: encoding_models.EncodedMlpModel, input_example: T, num_iterations: int = 1000,
                 epsilon:Union[float, Tuple[float, float]] = 1e-3, verbose=False, num_blocks: Optional[int] = None):
        self.num_samples = input_example.shape[0]
        self.verbose = verbose
        super(SpatialFixedController, self).__init__(model)
        input_example = input_example.clone().detach()
        in_progress = torch.ones(input_example.shape[0], dtype=torch.bool)
        self.register_buffer('input_example', input_example)
        self.register_buffer('in_progress', in_progress)
        if num_blocks is None:
            self.block_size = model.domain_dim
            self.num_blocks = (self.encoding_dim - self.block_size) // self.block_size
        else:
            self.num_blocks = num_blocks
            self.block_size = self.encoding_dim // num_blocks
        self.mask[:, self.block_size:] = 0
        self.cur_block = self.block_size
        self.next_block = self.block_size * 2
        self.block_iterations = 3 * num_iterations // (4 * self.num_blocks)
        self.progress_iterations = self.block_iterations * self.num_blocks
        self.trigger = False
        self.epsilon_ = epsilon
        self.buffer_size = self.block_iterations // 2
        log_buffer = torch.ones(self.buffer_size, self.num_samples, dtype=torch.bool)
        self.register_buffer('log_buffer', log_buffer)


class SpatialControllerStashed(ProgressiveEncoderController):

    @property
    def name(self):
        return 'spatial_progression_stashed'

    def stash_iteration(self, loss: T, ignore: bool = False, alphas_sec: Optional[T] = None):
        if not ignore:
            loss = loss.clone().detach()
            with torch.no_grad():
                inds, alphas = self.stash
                if alphas_sec is not None:
                    alphas = alphas * alphas_sec[:, None]
                loss = (loss[:, None] * alphas).flatten()
                inds = inds.flatten()
                self.log_buffer[inds] += loss
                self.log_counter[inds] += alphas.flatten()
        super(SpatialControllerStashed, self).stash_iteration(loss)

    def reset_buffer_(self):
        self.log_buffer[:] = 0
        self.log_counter[:] = 0
        self.iteration = 0

    @functools.lru_cache
    def get_blur_log_weight(self, k, device: D) -> T:
        w = torch.ones(1, 1, *([k] * self.mask_dim), device=device)
        return w

    @functools.lru_cache
    def get_mask_log_weight(self, k, device: D) -> T:
        w = torch.ones(k ** self.mask_dim, device=device)
        w = w / (k ** self.mask_dim - 1)
        w[k ** self.mask_dim // 2] = 0
        w = w.view(1, 1, *([k] * self.mask_dim))
        return w

    def convolove_log_(self, log_buffer: T, empty_log: T) -> T:
        mk = self.k // 2
        conv = [nnf.conv1d, nnf.conv2d, nnf.conv3d][self.mask_dim -1]
        log_buffer = log_buffer.view(1, 1, *([self.res] * self.mask_dim))
        w = self.get_blur_log_weight(self.k, log_buffer.device)
        if empty_log.any():
            empty_log = empty_log.view(1, 1, *([self.res] * self.mask_dim))
            w_ = self.get_mask_log_weight(self.k, log_buffer.device)
            log_buffer_ = nnf.pad(log_buffer, tuple([mk] * (self.mask_dim * 2)), 'replicate')
            log_buffer_ = conv(log_buffer_, w_, stride=1)
            log_buffer[empty_log] = log_buffer_[empty_log]
        log_buffer = nnf.pad(log_buffer, tuple([mk] * (self.mask_dim * 2)), 'replicate')
        log_buffer = conv(log_buffer, w / (self.k ** self.mask_dim), stride=1)
        log_buffer = log_buffer.flatten()
        return log_buffer

    def convolove_log(self, log_buffer: T, empty_log: T) -> T:
        if self.mask_dim < 4:
            return self.convolove_log_(log_buffer, empty_log)
        else:
            raise NotImplemented

    @property
    def not_visited_mask(self) -> T:
        return self.log_counter.eq(0)

    @property
    def visited_percent(self) -> float:
        not_visited = float(self.not_visited_mask.sum().item())
        return 1 - not_visited / float(self.log_counter.numel())

    def update_progress(self, ignore: bool = False):
        with torch.no_grad():
            if not ignore:
                empty_log = self.not_visited_mask
                self.log_counter[empty_log] = 1
                log_buffer = self.log_buffer / self.log_counter
                log_buffer = self.convolove_log(log_buffer, empty_log)
                self.in_progress = self.in_progress * log_buffer.gt(self.epsilon)
            self.increase_block()
            self.reset_buffer_()

    def is_full(self) -> bool:
        num_non_zero = self.log_counter.nonzero().shape[0]
        return num_non_zero == self.mask.shape[0]

    def increase_block(self):
        self.mask[self.in_progress, self.cur_block:  self.next_block] = 1
        self.cur_block = self.next_block
        self.next_block += self.block_size
        if self.model.encoding_dim - self.next_block < self.block_size:
            self.next_block = self.model.encoding_dim
        self.mask_ = None

    def update_mask(self):
        return
        if self.train() and self.iteration < self.block_iterations and self.in_progress.any():
            alpha = min(1., float(2 * (self.iteration % self.block_iterations)) / self.block_iterations)
            self.mask[self.in_progress, self.cur_block:  self.next_block] = alpha
            self.mask_ = None

    def init_mask(self):
        return torch.ones(self.res ** self.mask_dim, self.model.encoding_dim)

    def get_mask_(self) -> T:
        mk = self.k // 2
        conv = [nnf.conv1d, nnf.conv2d, nnf.conv3d][self.mask_dim - 1]
        mask = self.mask.permute(1, 0)
        mask = mask.view(-1, 1, *([self.res] * self.mask_dim))
        w = self.get_blur_log_weight(self.k, mask.device)
        mask = nnf.pad(mask, tuple([mk] * (self.mask_dim * 2)), 'replicate')
        mask = conv(mask, w / (self.k ** self.mask_dim), stride=1)
        mask = mask.view(mask.shape[0], -1).permute(1, 0)
        return mask

    def get_mask(self) -> T:
        if self.mask_ is None:
            self.mask_ = self.get_mask_()
            # self.mask_ = self.mask
        return self.mask_

    def flat_inds(self, indices: List[TS], alphas: List[TS]) -> TS:
        # alphas = self.fix_alphas(alphas)
        format_string = lambda x: ("{" f"0:0{self.mask_dim}b" + "}").format(x)
        mask_inds, mask_alphas = [], []
        for i in range(2 ** self.mask_dim):
            select = format_string(i)
            cur_ind, cur_alpha = 0, 1
            for j, (s, inds, alpha) in enumerate(zip(select, indices, alphas)):
                s = int(s)
                cur_ind += inds[s] * self.res ** j
                cur_alpha *= alpha[s]
            mask_inds.append(cur_ind.long())
            mask_alphas.append(cur_alpha)
        mask_inds = torch.stack(mask_inds, 1)
        mask_alphas = torch.stack(mask_alphas, 1)
        return mask_inds, mask_alphas

    def scale_dummy(self, x: T):
        return x

    def scale_real(self, x: T) -> T:
        x = (x - self.center_scale[0]) * self.center_scale[1]
        return x

    def set_scale(self, training_points: T):
        max_vals, min_vals = training_points.max(0)[0], training_points.min(0)[0]
        self.center_scale[0, 0] = ((max_vals + min_vals) / 2).to(self.center_scale.device)
        self.center_scale[1, 0] = (2 / (max_vals - min_vals)).to(self.center_scale.device)
        self.scale = self.scale_real

    def to(self, device):
        self.center_scale = self.center_scale.to(device)
        return super(SpatialControllerStashed, self).to(device)

    def interpolate_(self, x: T) -> T:
        x = self.scale(x)
        x_ = ((x + 1) / 2) * max((self.res - 2), 1) + .5
        inds = [(torch.floor(x_[:, i]), torch.ceil(x_[:, i] + 1e-6)) for i in range(self.mask_dim)]
        alphas = [(inds[i][1] - x_[:, i], x_[:, i] - inds[i][0]) for i in range(self.mask_dim)]
        inds, alphas = self.flat_inds(inds, alphas)
        self.stash = inds, alphas
        mask = self.get_mask()[inds]
        mask = torch.einsum('ndf,nd->nf', mask, alphas)
        return mask

    def interpolate(self, x) -> T:
        with torch.no_grad():
           return self.interpolate_(x)

    def forward(self, x: T, **kwargs):
        if 'mask_by' in kwargs:
            mask = self.interpolate(kwargs['mask_by'])
        else:
            mask = self.interpolate(x)
        if 'expand_by' in kwargs:
            mask = kwargs['expand_by'](mask, x)
        out = self.model(x, override_mask=mask)
        if 'get_mask' in kwargs:
            return out, mask
        return out

    def __init__(self, params: encoding_models.ModelParams, encoding_type: EncodingType, control_params: ControlParams):
        self.res = max(control_params.res, 3)
        self.mask_dim = params.domain_dim if control_params.mask_dim is None else control_params.mask_dim
        super(SpatialControllerStashed, self).__init__(params, encoding_type, control_params)
        in_progress = torch.ones(*self.mask.shape[:-1], dtype=torch.bool)
        self.register_buffer('in_progress', in_progress)

        if control_params.num_blocks is None:
            self.block_size = self.model.domain_dim
            control_params.num_blocks = (self.encoding_dim - self.block_size) // self.block_size
        else:
            self.block_size = self.encoding_dim // control_params.num_blocks
        self.mask[:, self.block_size:] = 0
        self.mask_: Optional[T] = None
        self.cur_block = self.block_size
        self.next_block = self.block_size * 2
        self.block_iterations = int(control_params.num_iterations / (2 * control_params.num_blocks))
        self.progress_iterations = self.block_iterations * control_params.num_blocks
        self.trigger = False
        self.epsilon = control_params.epsilon
        self.k = 5 if self.mask.shape[0] > 100 else 3
        self.stash: TNS = None, None
        log_buffer = torch.zeros(self.mask.shape[0], dtype=torch.float)
        log_counter = torch.zeros(self.mask.shape[0], dtype=torch.float)
        self.register_buffer('log_buffer', log_buffer.detach())
        self.register_buffer('log_counter', log_counter.detach())
        self.center_scale = torch.zeros(2, 1, self.domain_dim)
        self.center_scale[1, :] = 1
        self.scale = self.scale_dummy


def get_controlled_model(model_params: encoding_models.ModelParams, encoding_type: EncodingType,
                         control_params: ControlParams, control_type: ControllerType) -> EncodedController:
    if encoding_type is EncodingType.NoEnc:
        control_type = ControllerType.NoControl
    if control_type is ControllerType.NoControl:
        return NoControl(model_params, encoding_type, control_params)
    elif control_type is ControllerType.GlobalProgression:
        model_params.use_id_encoding = True
        return GlobalProgressionControllerEarly(model_params, encoding_type, control_params)
    elif control_type is ControllerType.SpatialProgressionStashed:
        model_params.use_id_encoding = True
        return SpatialControllerStashed(model_params, encoding_type, control_params)
    else:
        raise ValueError(f"{encoding_type.value} is not supported")