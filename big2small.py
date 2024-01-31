import os
import torch
from tqdm import tqdm

def big2small(folder):
    origin_videos = os.path.join(folder, "origin.pt")
    torch.save(torch.load(origin_videos).clone(), origin_videos)

    for i in tqdm(range(100)):
        result_videos = os.path.join(folder, f"result_{i}.pt")
        torch.save(torch.load(result_videos).clone(), result_videos)

    # best_videos = os.path.join(folder, "result_best.pt")
    # torch.save(torch.load(best_videos).clone(), best_videos)

    print(folder, "done")

# big2small("./logs_validation/pretrained_diffusion/BAIR/bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada_flowdiff_best_239.058_2000_100")
# big2small("./logs_validation/pretrained_diffusion/BAIR/bair64_DM_Batch66_lr2.e-4_c2p7_STW_adaptor_multi_traj_ada_flowdiff_0066_S095000_9000_100")
# big2small("./logs_validation/pretrained_diffusion/BAIR/bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume_flowdiff_0064_S190000_9000_100")
# big2small("./logs_validation/pretrained_diffusion/BAIR/bair64_DM_Batch64_lr2e-4_c2p4_STW_adaptor_scale0.50_multi_traj_flowdiff_best_315.362_10000_100")

# big2small("./logs_validation/pretrained_diffusion/KTH/kth64_DM_Batch32_lr2e-4_c10p4_STW_adaptor_scale0.50_multi_traj_ada_flowdiff_best_355.236_1000_100")
# big2small("./logs_validation/pretrained_diffusion/KTH/kth64_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada_flowdiff_0032_S098000_7000_100")

# big2small("./logs_validation/pretrained_diffusion_old/Cityscapes/cityscapes128_DM_Batch40_lr1.5e-4_c2p5_STW_adaptor_scale0.25_multi_traj_ada_flowdiff_best_181.577_1234_50")
big2small("./logs_validation/pretrained_diffusion_old/Cityscapes/cityscapes128_DM_Batch40_lr1.5e-4_c2p7_STW_adaptor_scale0.25_multi_traj_ada_flowdiff_best_1234_50")

# 45/256炸一次
# 4：22  2/256 26392 45.6
# 4：35  9/256 31054 46.0
# 4：40 13/256 32596 46.1
# 4：46 17/256 34852 46.3
# 5：00 26/256 39928 46.6
# 5：10 32/256 43312 46.3



# 5.11 0  25414 45.3
# 5.17 3  25264 46.8
# 5.21 6  25414 48.3
# 5.36 16 25414 58.1
# 6.49 64 25264 81.0
# 12.00

98