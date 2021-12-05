from custom_types import *
import abc


class ModelParams:

    def fill_args(self, **kwargs):
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)

    def __init__(self, **kwargs):
        self.domain_dim = 2
        self.num_frequencies = 256
        self.std = 20
        self.num_layers = 3
        self.hidden_dim = 256
        self.output_channels = 3
        self.use_id_encoding = False
        self.fill_args(**kwargs)


class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, layers: Union[List[int], Tuple[int, ...]]):
        super(MLP, self).__init__()
        layers_ = []
        for i in range(len(layers) - 1):
            layers_.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layers_.append(nn.ReLU(True))
        self.model = nn.Sequential(*layers_)


class EncodingLayer(nn.Module, abc.ABC):

    @property
    @abc.abstractmethod
    def output_channels(self) -> int:
        raise NotImplemented


class EncodedMlpModel(nn.Module, abc.ABC):

    @property
    def domain_dim(self):
        return self.opt.domain_dim

    @property
    def encoding_dim(self):
        if self.opt.use_id_encoding:
            return self.encode.output_channels + self.opt.domain_dim
        else:
            return self.encode.output_channels

    @abc.abstractmethod
    def get_encoding_layer(self) -> EncodingLayer:
        raise NotImplemented

    def get_mlp_model(self) -> nn.Module:
        dims = [self.encoding_dim] + self.opt.num_layers * [self.opt.hidden_dim] + [self.opt.output_channels]
        return MLP(dims)

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        encoding = self.encode(x)
        if self.opt.use_id_encoding:
            return torch.cat((x, encoding), dim=-1)
        else:
            return encoding

    def forward(self, x: T, *args, **kwargs) -> T:
        base_code = self.get_encoding(x)
        if 'override_mask' in kwargs:
            base_code = base_code * kwargs['override_mask']
        out = self.mlp_forward(base_code)
        return out

    def __init__(self, opt: ModelParams):
        super(EncodedMlpModel, self).__init__()
        self.opt = opt
        self.encode = self.get_encoding_layer()
        self.model = self.get_mlp_model()


class IdEncoding(EncodingLayer):

    def forward(self, x):
        return x

    @property
    def output_channels(self) -> int:
        return self.domain_dim

    def __init__(self, domain_dim: int):
        super(IdEncoding, self).__init__()
        self.domain_dim = domain_dim


class BaseModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return IdEncoding(self.opt.domain_dim)


class FourierFeatures(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies * 2

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x = x * 2 * np.pi
        out = torch.matmul(x, self.frequencies)
        out = torch.sin(out), torch.cos(out)
        out = torch.stack(out, dim=2).view(*shape, self.output_channels)
        return out

    @abc.abstractmethod
    def init_frequencies(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(FourierFeatures, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies: int = num_frequencies
        frequencies = self.init_frequencies(std)
        self.register_buffer("frequencies", frequencies)


class GaussianRandomFourierFeatures(FourierFeatures):

    def init_frequencies(self, std: float) -> T:
        magnitude = torch.randn(self.num_frequencies) * std
        order = magnitude.abs().argsort(0)
        magnitude = magnitude[order]
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies)
        frequencies = nnf.normalize(frequencies, p=2, dim=0) * magnitude[None, :]
        return frequencies


class PositionalEncoding(EncodingLayer):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies * self.domain_dim * 2

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        out: T = torch.einsum('f,nd->nfd', self.frequencies, x)
        out = torch.cat((torch.cos(out), torch.sin(out)), dim=2).view(-1, self.output_channels)
        return out.view(*shape, -1)

    def __init__(self, domain_dim: int, num_frequencies: int):
        super(PositionalEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies
        frequencies = torch.tensor([2. ** i * np.pi for i in range(num_frequencies)])
        self.register_buffer("frequencies", frequencies)


class RadialBasisEncoding(EncodingLayer):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        out = (x[:, None, :] - self.centres[None, :, :]).pow(2).sum(2)
        out = out * self.sigma[None, :] ** 2
        out = torch.exp(-out)
        return out.view(*shape, -1)

    def __init__(self, domain_dim: int, num_frequencies: int, std: int):
        super(RadialBasisEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies * 2
        centres = torch.rand(self.num_frequencies, domain_dim) * 2 - 1
        sigma = (torch.randn(self.num_frequencies).abs() * std + 1)
        sigma = sigma.sort()[0]
        self.register_buffer("centres", centres)
        self.register_buffer("sigma", sigma)


class PeriodicRadialBasisEncoding(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x_a = x[:, None, :] + self.offsets[None, :]  # n f d
        x_b = x_a + (1 / self.sigma[None, :, None])  # n f d
        out = torch.stack((x_a, x_b), dim=2)
        out = (out % (2 / self.sigma[None, :, None, None])) * 2 - (2 / self.sigma[None, :, None, None])
        out = out.pow(2).sum(3)  # n f 2
        out = out * self.sigma[None, :, None] ** 2
        out = out.view(-1, self.output_channels)
        out = torch.exp(-out) * 2 - 1
        return out.view(*shape, self.output_channels)

    @abc.abstractmethod
    def init_frequencies(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(PeriodicRadialBasisEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies
        sigma = self.init_frequencies(std)
        offsets = (torch.rand(self.num_frequencies, domain_dim) * 2 - 1) % (2 / sigma[:, None])
        sigma = sigma.sort()[0]
        self.register_buffer("offsets", offsets)
        self.register_buffer("sigma", sigma)


class RandomRadialBasisGridEncoding(PeriodicRadialBasisEncoding):

    def init_frequencies(self, std: float) -> T:
        return torch.randn(self.num_frequencies).abs() * std + 1


class UniformRadialBasisGridEncoding(PeriodicRadialBasisEncoding):

    def init_frequencies(self, std: float) -> T:
        frequencies = torch.linspace(0, std * np.sqrt(3), self.num_frequencies)
        frequencies = frequencies + frequencies[1] / 2
        return frequencies


class FFModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return GaussianRandomFourierFeatures(self.opt.domain_dim, self.opt.num_frequencies, self.opt.std)


class PEModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return PositionalEncoding(self.opt.domain_dim, self.opt.num_frequencies)


class RbfModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return RadialBasisEncoding(self.opt.domain_dim, self.opt.num_frequencies, self.opt.std)


class PrbfModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return UniformRadialBasisGridEncoding(self.opt.domain_dim, self.opt.num_frequencies, self.opt.std)


def get_model(params: ModelParams, model_type: EncodingType) -> EncodedMlpModel:
    if model_type is EncodingType.FF:
        model = FFModel(params)
    elif model_type is EncodingType.NoEnc:
        model = BaseModel(params)
    elif model_type == EncodingType.RBF:
        model = RbfModel(params)
    elif model_type == EncodingType.PRBF:
        model = PrbfModel(params)
    elif model_type is EncodingType.PE:
        model = PEModel(params)
    else:
        raise ValueError(f"{model_type.value} is not supported")
    return model