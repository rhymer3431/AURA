import torch
import torch.nn as nn
import math

from .depth_anything_encoder import build_depth_anything_v2_encoder


def _to_device_float_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32).contiguous()
    return torch.as_tensor(value, dtype=torch.float32, device=device).contiguous()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding

class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 memory_size=8,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.rgb_model = build_depth_anything_v2_encoder("vits").float()
        self.rgb_model.eval()
        self.register_buffer(
            "preprocess_mean",
            torch.tensor([0.485,0.456,0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "preprocess_std",
            torch.tensor([0.229,0.224,0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.depth_model = build_depth_anything_v2_encoder("vits").float()
        self.depth_model.eval()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size*16)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*256)
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)

    def forward(self,images,depths):
        with torch.no_grad():
            device = self.preprocess_mean.device
            tensor_images = _to_device_float_tensor(images, device)
            tensor_depths = _to_device_float_tensor(depths, device)
            if tensor_images.ndim == 4:
                tensor_images = tensor_images.permute(0,3,1,2)
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean) / self.preprocess_std
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            elif tensor_images.ndim == 5:
                tensor_images = tensor_images.permute(0,1,4,2,3)
                B,T,C,H,W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean) / self.preprocess_std
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*256,-1)
            else:
                raise ValueError(f"images must be rank-4 or rank-5, got {tuple(tensor_images.shape)}")
            if tensor_depths.ndim == 4:
                tensor_depths = tensor_depths.permute(0,3,1,2)
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            elif tensor_depths.ndim == 5:
                tensor_depths = tensor_depths.permute(0,1,4,2,3)
                B,T,C,H,W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B,T*256,-1)
            else:
                raise ValueError(f"depths must be rank-4 or rank-5, got {tuple(tensor_depths.shape)}")
            former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
            former_query = self.former_query(image_token.new_zeros((image_token.shape[0], self.memory_size * 16, 384)))
            memory_token = self.former_net(former_query,former_token)
            memory_token = self.project_layer(memory_token)
            return memory_token

class NavDP_ImageGoal_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        self.imagegoal_encoder = build_depth_anything_v2_encoder("vits").float()
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(in_channels=6,
                                                            out_channels = self.imagegoal_encoder.patch_embed.proj.out_channels,
                                                            kernel_size = self.imagegoal_encoder.patch_embed.proj.kernel_size,
                                                            stride = self.imagegoal_encoder.patch_embed.proj.stride,
                                                            padding = self.imagegoal_encoder.patch_embed.proj.padding)
        self.imagegoal_encoder.eval()
        self.project_layer = nn.Linear(384,embed_size)

    def forward(self,images):
        with torch.no_grad():
            assert len(images.shape) == 4 # B,C,H,W
            device = self.project_layer.weight.device
            tensor_images = _to_device_float_tensor(images, device).permute(0,3,1,2)
            image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
            image_token = self.project_layer(image_token)
            return image_token

class NavDP_PixelGoal_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        self.pixelgoal_encoder = build_depth_anything_v2_encoder("vits").float()
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(in_channels=4,
                                                            out_channels = self.pixelgoal_encoder.patch_embed.proj.out_channels,
                                                            kernel_size = self.pixelgoal_encoder.patch_embed.proj.kernel_size,
                                                            stride = self.pixelgoal_encoder.patch_embed.proj.stride,
                                                            padding = self.pixelgoal_encoder.patch_embed.proj.padding)
        self.pixelgoal_encoder.eval()
        self.project_layer = nn.Linear(384,embed_size)

    def forward(self,images):
        with torch.no_grad():
            assert len(images.shape) == 4 # B,C,H,W
            device = self.project_layer.weight.device
            tensor_images = _to_device_float_tensor(images, device).permute(0,3,1,2)
            image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
            image_token = self.project_layer(image_token)
            return image_token

if __name__ == "__main__":
    backbone = NavDP_PixelGoal_Backbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,4)
    print(backbone(images).shape)

    backbone = NavDP_ImageGoal_Backbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,6)
    print(backbone(images).shape)
