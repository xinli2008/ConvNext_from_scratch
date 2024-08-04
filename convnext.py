import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models._registry import register_model

class Block(nn.Module):
    r""" ConvNext Block. There are two equivalent implementations:
    (1) DWConv -> LayerNorm(channels_first) -> 1x1 Conv -> 1x1 Conv; 
    (2) DWConv -> Permute to (N, H, W, C); LayerNorm(channels_last) -> Linear -> GELU -> Linear; Permute back
    
    Args:
        dim(int): number of input channels.
        drop_path(float): stochastic depth rate. Default: 0.0
        layer_scale_init_value(float): init value for layer scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path = 0., layer_scale_init_value = 1e-6):
        super(Block, self).__init__()

        # -------------------------------------------------------------------------------
        # 这里的groups的作用是什么？
        # 在pytorch中的nn.conv2d中, groups参数决定了卷积层的分组形式。具体来说, 他控制输入通道和输出通道之间的连接方式。
        # 1、默认值为1：
        # 2、
        self.dwconv = nn.Conv2d(dim, dim, kernel_size = 7, padding = 3, groups = dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps = 1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad = True) if layer_scale_init_value > 0 else None
        
        # -------------------------------------------------------------------------------
        # Dropout和Dropout的区别和联系是什么？
        # 首先, dropout和droppath都是正则化技术, 用来防止神经网络过拟合, 但是他们在实现和应用方式上有一些区别。
        # Dropout: 
        # - dropout的基本思想是在训练过程中随机"丢弃"一部分神经元。具体来说, 就是每个神经元, 他都有一定的概率暂时从网络中移除。
        # 这种方法可以有效的防止网络的某些部分过分依赖特定的神经元, 从而提高网络的泛化能力。
        # DropPath:
        # - droppath的基本思想是随机"丢弃"整个残差块或路径, 而不仅仅是单个神经元。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()
        x = input + self.drop_path(x)
        return x
    

class ConvNeXt(nn.Module):
    r""" ConvNeXt
    Args:
        in_channels(int):   Number of input image channels. Default: 3
        num_classes(int):   Number of classes for classification head. Default: 1000
        depths(list):       Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims(list)          Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate(float): stochastic depth rate. Default: 0.
        layer_scale_init_value(float): init value for layer scale. Default: 1e-6.
        head_init_scale(float): init scaling value for classifier weights and bias. Default: 1
    """
    def __init__(self, in_channels = 3, num_classes = 1000, depths = [3, 3, 9, 3], 
                dims = [96, 192, 384, 768], drop_path_rate = 0.01, 
                layer_scale_init_value = 1e-6, head_init_scale = 1):
        super(ConvNeXt, self).__init__()

        self.downsamples_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size = 4, stride = 4),
            LayerNorm(dims[0], eps = 1e-6, data_format = "channels_first")
        )
        self.downsamples_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps = 1e-6, data_format = "channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size = 2, stride = 2)
            )
            self.downsamples_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList()   # 4 feature resolution stages, each consisting of multiple residual blocks
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            # -------------------------------------------------------------------------------
            # 1、这里为什么需要用一个*呢？
            # nn.sequential是一个顺序容器。模块将按照它们在传递给sequential的顺序依次被添加到计算图中。
            # 通过使用*解包操作符, 列表中的每个block示例都会添加到nn.sequential中, 从而形成一个顺序的模块序列。相当于：
            # stage = nn.Sequential(
            # *[
            #     Block(dim=192, drop_path=drop_rates[5], layer_scale_init_value=1e-6),
            #     Block(dim=192, drop_path=drop_rates[6], layer_scale_init_value=1e-6),
            #     Block(dim=192, drop_path=drop_rates[7], layer_scale_init_value=1e-6)
            # ])
            # 2、*和**解包有什么区别吗？
            # 首先, *和**都是解包（unpacking）的操作符, 但是他们用于不同的场景和对象类型。
            # *用于列表、元祖等可迭代对象, **用于字典
            stage = nn.Sequential(
                *[Block(dim = dims[i], drop_path = drop_rates[cur + j], layer_scale_init_value = layer_scale_init_value)
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps = 1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # init weight
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # --------------------------------------------------------------------------------
            # trunc_normal_(m.weight, std = .02): 对权重进行截断正态分布初始化, 标准差为0.02
            # 这是一种初始化方法, 可以确保权重不会有太大的值, 从而有助于模型可以稳定训练。
            # constant_：将偏置初始化为常数 0。
            trunc_normal_(m.weight, std = 0.02) 
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsamples_layers[i](x)
            x = self.stages[i](x)
        # -------------------------------------------------------------------------------
        # x.mean([-2, -1])是什么意思？
        # x.mean([-2, -1])执行的是全局平均池化操作。
        # 假如它的tensor的shape是[b, c, h, w], 则x.mean([-2, -1])是在特征图的高度和宽度这两个维度上计算平均值。
        # 它将每个通道上的所有像素值平均化, 从而将[b, c, h, w]转化为[b, c]
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == "__main__":
    model = convnext_tiny()
    input_tensor = torch.randn([1, 3, 224, 224])
    output_tensor = model(input_tensor)
    pass