# use diffusion model to generate pseudo ground truth flow volume based on RegionMM
# 3D noise to 3D flow
# flow size: 2*32*32*40
# some codes based on https://github.com/lucidrains/video-diffusion-pytorch

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LFAE.generator import Generator
from model.LFAE.bg_motion_predictor import BGMotionPredictor
from model.LFAE.region_predictor import RegionPredictor
from model.DM.video_flow_diffusion import Unet3D, GaussianDiffusion
import yaml


class FlowDiffusion(nn.Module):
    def __init__(self, 
            config="",
            pretrained_pth="",
            is_train=True,
            ddim_sampling_eta=1., 
            timesteps=1000,
            dim_mults=(1, 2, 4, 8),
            learn_null_cond=False,
            use_deconv=True,
            padding_mode="zeros",
        ):
        super(FlowDiffusion, self).__init__()
        
        flow_params = config['flow_params']['model_params']
        diffusion_params = config['diffusion_params']['model_params']
        dataset_params = config['dataset_params']

        self.use_residual_flow = diffusion_params['use_residual_flow']
        self.only_use_flow = diffusion_params['only_use_flow']

        if pretrained_pth != "":
            checkpoint = torch.load(pretrained_pth)

        self.generator = Generator(num_regions=flow_params['num_regions'],
                                   num_channels=flow_params['num_channels'],
                                   revert_axis_swap=flow_params['revert_axis_swap'],
                                   **flow_params['generator_params']).cuda()
        if pretrained_pth != "":
            self.generator.load_state_dict(checkpoint['generator'])
            self.generator.eval()
            self.set_requires_grad(self.generator, False)

        self.region_predictor = RegionPredictor(num_regions=flow_params['num_regions'],
                                                num_channels=flow_params['num_channels'],
                                                estimate_affine=flow_params['estimate_affine'],
                                                **flow_params['region_predictor_params']).cuda()
        if pretrained_pth != "":
            self.region_predictor.load_state_dict(checkpoint['region_predictor'])
            self.region_predictor.eval()
            self.set_requires_grad(self.region_predictor, False)

        self.bg_predictor = BGMotionPredictor(num_channels=flow_params['num_channels'],
                                              **flow_params['bg_predictor_params'])
        if pretrained_pth != "":
            self.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
            self.bg_predictor.eval()
            self.set_requires_grad(self.bg_predictor, False)

        self.unet = Unet3D(
            dim=64,
            channels=3 + 256,
            out_grid_dim=2,
            out_conf_dim=1,
            dim_mults=dim_mults,
            use_bert_text_cond=False,
            learn_null_cond=learn_null_cond,
            use_final_activation=False,
            use_deconv=use_deconv,
            padding_mode=padding_mode,
            cond_num=dataset_params['train_params']['cond_frames'],
            pred_num=dataset_params['train_params']['pred_frames']
        )

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=dataset_params['frame_shape'],
            num_frames=dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames'],
            sampling_timesteps=diffusion_params['sampling_timesteps'],
            timesteps=timesteps,  # number of steps
            loss_type=diffusion_params['loss_type'],  # L1 or L2
            use_dynamic_thres=True,
            null_cond_prob=diffusion_params['null_cond_prob'],
            ddim_sampling_eta=ddim_sampling_eta,
        )

        self.cond_frame_num = dataset_params['train_params']['cond_frames']
        self.pred_frame_num = dataset_params['train_params']['pred_frames']
        self.frame_num = self.cond_frame_num + self.pred_frame_num

        # self.ref_img = None
        # self.ref_img_fea = None
        # self.real_vid = None
        # self.real_out_vid = None
        # self.real_warped_vid = None
        # self.real_vid_grid = None
        # self.real_vid_conf = None

        # self.fake_out_vid = None
        # self.fake_warped_vid = None
        # self.fake_vid_grid = None
        # self.fake_vid_conf = None

        # self.sample_out_vid = None
        # self.sample_warped_vid = None
        # self.sample_vid_grid = None
        # self.sample_vid_conf = None

        # training
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()
            # self.lr = lr
            # self.loss = torch.tensor(0.0).cuda()
            # self.rec_loss = torch.tensor(0.0).cuda()
            # self.rec_warp_loss = torch.tensor(0.0).cuda()
            # self.optimizer_diff = torch.optim.Adam(self.diffusion.parameters(),
            #                                        lr=lr, betas=adam_betas)

    def forward(self, real_vid):
        # compute pseudo ground-truth flow
        b, _, nf, H, W = real_vid.size()
        ret = {}
        loss = torch.tensor(0.0).cuda()

        real_grid_list = []
        real_conf_list = []
        real_out_img_list = []
        real_warped_img_list = []
        with torch.no_grad():
            ref_img = real_vid[:,:,self.cond_frame_num-1,:,:]
            source_region_params = self.region_predictor(ref_img)
            for idx in range(nf):
                driving_region_params = self.region_predictor(real_vid[:, :, idx, :, :])
                bg_params = self.bg_predictor(ref_img, real_vid[:, :, idx, :, :])
                generated = self.generator(ref_img, source_region_params=source_region_params,
                                           driving_region_params=driving_region_params, bg_params=bg_params)
                generated.update({'source_region_params': source_region_params,
                                  'driving_region_params': driving_region_params})
                real_grid_list.append(generated["optical_flow"].permute(0, 3, 1, 2))
                # normalized occlusion map
                real_conf_list.append(generated["occlusion_map"])
                real_out_img_list.append(generated["prediction"])
                real_warped_img_list.append(generated["deformed"])
        
        if self.is_train:
            # pred frames
            pred_frames = real_vid[:,:,self.cond_frame_num : self.cond_frame_num+self.pred_frame_num]
        del real_vid
        
        torch.cuda.empty_cache()

        real_vid_grid = torch.stack(real_grid_list, dim=2)
        real_vid_conf = torch.stack(real_conf_list, dim=2)
        real_out_vid = torch.stack(real_out_img_list, dim=2)
        real_warped_vid = torch.stack(real_warped_img_list, dim=2)
        # reference images are the same for different time steps, just pick the final one
        ref_img_fea = generated["bottle_neck_feat"].detach()
        ret['real_vid_grid'] = real_vid_grid
        ret['real_vid_conf'] = real_vid_conf
        ret['real_out_vid'] = real_out_vid
        ret['real_warped_vid'] = real_warped_vid
        ret['ref_img_fea'] = ref_img_fea
        
        if self.is_train:
            if self.use_residual_flow:
                h, w = H//4, W//4
                identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
                frames = torch.cat((real_vid_grid-identity_grid, real_vid_conf*2-1), dim=1)
            else:
                frames = torch.cat((real_vid_grid, real_vid_conf*2-1), dim=1)

            loss, pred = self.diffusion(frames[:,:,:self.cond_frame_num], frames[:,:,self.cond_frame_num:self.cond_frame_num+self.pred_frame_num], ref_img_fea)
            ret['loss'] = loss
            
            with torch.no_grad():
                fake_out_img_list = []
                fake_warped_img_list = []
                # pred = self.diffusion.pred_x0
                if self.use_residual_flow:
                    fake_vid_grid = pred[:, :2, :, :, :] + identity_grid
                else:
                    fake_vid_grid = pred[:, :2, :, :, :]
                fake_vid_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
                for idx in range(self.pred_frame_num):
                    fake_grid = fake_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
                    fake_conf = fake_vid_conf[:, :, idx, :, :]
                    # predict fake out image and fake warped image
                    generated = self.generator.forward_with_flow(source_image=ref_img, optical_flow=fake_grid, occlusion_map=fake_conf)
                    fake_out_img_list.append(generated["prediction"])
                    fake_warped_img_list.append(generated["deformed"])
                fake_out_vid = torch.stack(fake_out_img_list, dim=2)
                fake_warped_vid = torch.stack(fake_warped_img_list, dim=2)
                rec_loss = nn.L1Loss()(pred_frames, fake_out_vid)
                rec_warp_loss = nn.L1Loss()(pred_frames, fake_warped_vid)
                ret['fake_vid_grid'] = fake_vid_grid
                ret['fake_vid_conf'] = fake_vid_conf
                ret['fake_out_vid'] = fake_out_vid
                ret['fake_warped_vid'] = fake_warped_vid
                ret['rec_loss'] = rec_loss
                ret['rec_warp_loss'] = rec_warp_loss
        return ret

    def sample_one_video(self, cond_scale):
        sample_img_fea = self.generator.compute_fea(self.sample_img)
        # if cond_scale = 1.0, not using unconditional model
        pred = self.diffusion.sample(self.sample_img_fea, cond=self.sample_text,
                                     batch_size=1, cond_scale=cond_scale)
        if self.use_residual_flow:
            b, _, nf, h, w = pred[:, :2, :, :, :].size()
            identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
            self.sample_vid_grid = pred[:, :2, :, :, :] + identity_grid
        else:
            self.sample_vid_grid = pred[:, :2, :, :, :]
        self.sample_vid_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
        nf = self.sample_vid_grid.size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(nf):
                sample_grid = self.sample_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
                sample_conf = self.sample_vid_conf[:, :, idx, :, :]
                # predict fake out image and fake warped image
                generated = self.generator.forward_with_flow(source_image=self.sample_img,
                                                             optical_flow=sample_grid,
                                                             occlusion_map=sample_conf)
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
        self.sample_out_vid = torch.stack(sample_out_img_list, dim=2)
        self.sample_warped_vid = torch.stack(sample_warped_img_list, dim=2)


    def print_learning_rate(self):
        lr = self.optimizer_diff.param_groups[0]['lr']
        assert lr > 0
        print('lr= %.7f' % lr)

    def get_grid(self, b, nf, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid.permute(0, 3, 1, 2).unsqueeze(dim=2).repeat(1, 1, nf, 1, 1)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    bs = 5
    img_size = 64
    num_frames = 40
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand((bs, 3, img_size, img_size), dtype=torch.float32)
    real_vid = torch.rand((bs, 3, num_frames, img_size, img_size), dtype=torch.float32)
    model = FlowDiffusion(use_residual_flow=False,
                          sampling_timesteps=10,
                          img_size=16,
                          config_pth="/workspace/code/CVPR23_LFDM/config/mug128.yaml",
                          pretrained_pth="")
    model.cuda()
    # model.train()
    # model.set_train_input(ref_img=ref_img, real_vid=real_vid, ref_text=ref_text)
    # model.optimize_parameters()
    model.eval()
    model.set_sample_input(sample_img=ref_img, sample_text=ref_text)
    model.sample_one_video(cond_scale=1.0)



