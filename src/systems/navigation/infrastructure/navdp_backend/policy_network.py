import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from .policy_backbone import *


def _to_runtime_float_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32).contiguous()
    return torch.as_tensor(value, dtype=torch.float32, device=device).contiguous()


class NavDP_Policy(nn.Module):
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim

        # input encoders
        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size,token_dim,memory_size=memory_size,device=device)
        self.point_encoder = nn.Linear(3,self.token_dim)
        self.pixel_encoder = NavDP_PixelGoal_Backbone(image_size,token_dim,device=device)
        self.image_encoder = NavDP_ImageGoal_Backbone(image_size,token_dim,device=device)

        # fusion layers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                                        nhead = heads,
                                                        dim_feedforward = 4 * token_dim,
                                                        activation = 'gelu',
                                                        batch_first = True,
                                                        norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer,
                                             num_layers = self.temporal_depth)

        self.input_embed = nn.Linear(3,token_dim) # encode the actions for denoise/critic
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4) # time,point,image,pixel,input
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=10,
                                       beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True,
                                       prediction_type='epsilon')

        tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        cond_critic_mask = torch.zeros((predict_size,4 + memory_size * 16))
        cond_critic_mask[:,0:4] = float('-inf')
        self.register_buffer("tgt_mask", tgt_mask, persistent=False)
        self.register_buffer("cond_critic_mask", cond_critic_mask, persistent=False)
        self.register_buffer("short_trajectory_scale", torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32), persistent=False)

    def _runtime_device(self):
        return self.action_head.weight.device

    def _as_runtime_tensor(self, value):
        return _to_runtime_float_tensor(value, self._runtime_device())

    def predict_noise(self,last_actions,timestep,goal_embed,rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self._runtime_device())).unsqueeze(1).tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask)
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_mix_noise(self,last_actions,timestep,goal_embeds,rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self._runtime_device())).unsqueeze(1).tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embeds[0],goal_embeds[1],goal_embeds[2],rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embeds[0],goal_embeds[1],goal_embeds[2],rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask)
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_critic(self,predict_trajectory,rgbd_embed):
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1) +  self.cond_pos_embed(torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1))
        critic_output = self.decoder(tgt = action_embeddings, memory = cond_embeddings, memory_mask = self.cond_critic_mask)
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output

    def predict_pointgoal_action(self,goal_point,input_images,input_depths,sample_num=16):
        with torch.no_grad():
            tensor_point_goal = self._as_runtime_tensor(goal_point)
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_point.shape[0], self.predict_size, 3), device=self._runtime_device())
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),pointgoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_point.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_point.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * self.short_trajectory_scale

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_point.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_point.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_imagegoal_action(self,goal_image,input_images,input_depths,sample_num=16):
        with torch.no_grad():
            goal_image = self._as_runtime_tensor(goal_image)
            input_images = self._as_runtime_tensor(input_images)
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            imagegoal_embed = self.image_encoder(torch.cat((goal_image,input_images[:,-1]),dim=-1)).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self._runtime_device())
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),imagegoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * self.short_trajectory_scale

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_pixelgoal_action(self,goal_image,input_images,input_depths,sample_num=16):
        with torch.no_grad():
            goal_image = self._as_runtime_tensor(goal_image)
            input_images = self._as_runtime_tensor(input_images)
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            pixelgoal_embed = self.pixel_encoder(torch.cat((goal_image[:,:,:,None],input_images[:,-1]),dim=-1)).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            pixelgoal_embed = torch.repeat_interleave(pixelgoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self._runtime_device())
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),pixelgoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * self.short_trajectory_scale

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_nogoal_action(self,input_images,input_depths,sample_num=16):
        with torch.no_grad():
            input_images = self._as_runtime_tensor(input_images)
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            nogoal_embed = torch.repeat_interleave(nogoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * input_images.shape[0], self.predict_size, 3), device=self._runtime_device())
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),nogoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(input_images.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(input_images.shape[0],sample_num,self.predict_size,3)

            #distance = all_trajectory[:,-1,0:2].square().sum(dim=-1).sqrt()
            #critic_values[torch.where(distance<0.5)[0]] = -10.0
            #all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            print(trajectory_length.shape,trajectory_length.max(),trajectory_length.min())
            critic_values[torch.where(trajectory_length<1.0)] -= 10.0

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            #import pdb
            #pdb.set_trace()

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_ip_action(self,goal_point,goal_image,input_images,input_depths,sample_num=16):
        with torch.no_grad():
            tensor_point_goal = self._as_runtime_tensor(goal_point)
            goal_image = self._as_runtime_tensor(goal_image)
            input_images = self._as_runtime_tensor(input_images)
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            imagegoal_embed = self.image_encoder(torch.cat((goal_image,input_images[:,-1]),dim=-1)).unsqueeze(1)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed,sample_num,dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self._runtime_device())
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_mix_noise(naction,k.unsqueeze(0),[imagegoal_embed,pointgoal_embed,imagegoal_embed],rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * self.short_trajectory_scale

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0], device=all_trajectory.device).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
