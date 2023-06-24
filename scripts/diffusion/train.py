def train(
        config, 
        dataset_params,
        train_params, 
        log_dir, 
        checkpoint,
        args.device_ids
    ):

    print(config)

    train_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['train_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames'],
        mean=(0.0, 0.0, 0.0)
    )
    
    valid_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['valid_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=dataset_params['valid_params']['cond_frames'] + dataset_params['valid_params']['pred_frames'], 
        mean=(0.0, 0.0, 0.0),
        total_videos=dataset_params['valid_params']['total_videos'],
    )

    # FIXME: 可以改用adamw试试
    optimizer = torch.optim.Adam(
        model.diffusion.parameters(), 
        lr=train_params['lr'], 
        betas=(0.9, 0.99)
    )

    start_epoch = 0
    start_step = 0

    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            print("=> loading checkpoint '{}'".format(checkpoint))
            checkpoint = torch.load(checkpoint)
            if config["set_start"]:
                start_step = int(math.ceil(checkpoint['example'] / train_params['batch_size']))
            model_ckpt = model.diffusion.state_dict()
            for name, _ in model_ckpt.items():
                model_ckpt[name].copy_(checkpoint['diffusion'][name])
            model.diffusion.load_state_dict(model_ckpt)
            print("=> loaded checkpoint '{}'".format(checkpoint))
            if "optimizer_diff" in list(checkpoint.keys()):
                optimizer_diff.load_state_dict(checkpoint['optimizer_diff'])
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))
    else:
        print("NO checkpoint found!")

    scheduler = MultiStepLR(optimizer_diff, config['diffusion_params']['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)

    
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    valid_dataloader = data.DataLoader(,
                                batch_size=args.valid_batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)

    diffusion = FlowDiffusion(
        config=config,
        pretrained_pth=args.checkpoint,
        is_train=True,
    )

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_warp = AverageMeter()

    cnt = 0
    epoch_cnt = start_epoch
    actual_step = start_step
    start_epoch = int(math.ceil((start_step * args.batch_size)/NUM_EXAMPLES_PER_EPOCH))
    
    final_step = config["num_step_per_epoch"] * train_params["max_epochs"]

    print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer_diff.param_groups[0]["lr"]))
    fvd_best = 1e5
    
    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, batch in enumerate(train_dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)

            real_vids, real_names = batch
            # use first frame of each video as reference frame

            real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')

            print(real_vids.shape)
            # torch.Size([bs, length, c, h, w])
            ref_imgs = real_vids[:, :, CONDITION_FRAMES-1, :, :].clone().detach()
            bs = real_vids.size(0)

            model.set_train_input(cond_frame_num=CONDITION_FRAMES, train_frame_num=PRED_TRAIN_FRAMES, tot_frame_num=CONDITION_FRAMES + PRED_TRAIN_FRAMES)
            ret = model.optimize_parameters(real_vids[:,:,:CONDITION_FRAMES + PRED_TRAIN_FRAMES].cuda(), optimizer_diff)

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            losses.update(ret['loss'], bs)
            losses_rec.update(ret['rec_loss'], bs)
            losses_warp.update(ret['rec_warp_loss'], bs)

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.7f} ({loss.avg:.7f})\t'
                      'loss_rec {loss_rec.val:.4f} ({loss_rec.avg:.4f})\t'
                      'loss_warp {loss_warp.val:.4f} ({loss_warp.avg:.4f})'
                    .format(
                    cnt, actual_step, args.final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_rec=losses_rec,
                    loss_warp=losses_warp,
                ))

            if actual_step % args.save_img_freq == 0:
                msk_size = ref_imgs.shape[-1]
                save_src_img = sample_img(ref_imgs)
                save_tar_img = sample_img(real_vids[:, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2, :, :])
                save_real_out_img = sample_img(ret['real_out_vid'][:, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2, :, :])
                save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2, :, :])
                save_fake_out_img = sample_img(ret['fake_out_vid'][:, :, PRED_TRAIN_FRAMES//2, :, :])
                save_fake_warp_img = sample_img(ret['fake_warped_vid'][:, :, PRED_TRAIN_FRAMES//2, :, :])
                save_real_grid = grid2fig(ret['real_vid_grid'][0, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_fake_grid = grid2fig(ret['fake_vid_grid'][0, :, PRED_TRAIN_FRAMES//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_real_conf = conf2fig(ret['real_vid_conf'][0, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2], img_size=INPUT_SIZE)
                save_fake_conf = conf2fig(ret['fake_vid_conf'][0, :, PRED_TRAIN_FRAMES//2], img_size=INPUT_SIZE)
                new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                              + '_' + format(real_names[0], "06d") + ".png"
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_vid_freq == 0:
                print("saving video...")
                # num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(CONDITION_FRAMES + PRED_TRAIN_FRAMES):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_real_out_img = sample_img(ret['real_out_vid'][:, :, nf, :, :])
                    save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, nf, :, :])
                    save_fake_out_img = sample_img(ret['fake_out_vid'][:, :, nf - CONDITION_FRAMES, :, :])
                    save_fake_warp_img = sample_img(ret['fake_warped_vid'][:, :, nf - CONDITION_FRAMES, :, :])
                    save_real_grid = grid2fig(
                        ret['real_vid_grid'][0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_fake_grid = grid2fig(
                        ret['fake_vid_grid'][0, :, nf - CONDITION_FRAMES].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_real_conf = conf2fig(ret['real_vid_conf'][0, :, nf], img_size=INPUT_SIZE)
                    save_fake_conf = conf2fig(ret['fake_vid_conf'][0, :, nf - CONDITION_FRAMES], img_size=INPUT_SIZE)
                    new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                    new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                    new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                    new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                    new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                    new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                + '_' + format(real_names[0], "06d") + ".gif"
                new_vid_file = os.path.join(VIDSHOT_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)

            # sampling
            if actual_step % args.sample_vid_freq == 0:#  and cnt != 0
                print("sampling video...")
                model.set_sample_input(cond_frame_num=CONDITION_FRAMES, tot_frame_num=CONDITION_FRAMES + PRED_TEST_FRAMES)
                ret = model.sample_one_video(cond_scale=1.0, real_vid=real_vids.cuda())
                # num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(CONDITION_FRAMES + PRED_TRAIN_FRAMES):
                    save_tar_img = sample_img(real_vids[:, :, nf , :, :])
                    save_real_out_img = sample_img(ret['real_out_vid'][:, :, nf, :, :])
                    save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, nf, :, :])
                    save_sample_out_img = sample_img(ret['sample_out_vid'][:, :, nf, :, :])
                    save_sample_warp_img = sample_img(ret['sample_warped_vid'][:, :, nf, :, :])
                    save_real_grid = grid2fig(
                        ret['real_vid_grid'][0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_fake_grid = grid2fig(
                        ret['sample_vid_grid'][0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_real_conf = conf2fig(ret['real_vid_conf'][0, :, nf], img_size=INPUT_SIZE)
                    save_fake_conf = conf2fig(ret['sample_vid_conf'][0, :, nf], img_size=INPUT_SIZE)
                    new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                    new_im.paste(Image.fromarray(save_sample_out_img, 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(save_sample_warp_img, 'RGB'), (msk_size * 2, msk_size))
                    new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                    new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                    new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                    new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                + '_' + format(real_names[0], "06d") + ".gif"
                new_vid_file = os.path.join(SAMPLE_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)
            # if actual_step % args.sample_vid_freq == 0:

            # save model at i-th step
            if actual_step % args.save_pred_every == 0 and cnt != 0:
                # run validation
                fvd = valid(valid_dataloader=valid_dataloader, model=model, epoch=epoch_cnt)
                if fvd<fvd_best:
                    fvd_best = fvd
                    epoch_best = epoch_cnt
                print('=====================================')
                print('history best fvd:', fvd_best)
                print('history best epoch:', epoch_best)
                print('=====================================')
                # save model
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.diffusion.state_dict(),
                            'optimizer_diff': optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            # update saved model
            if actual_step % args.update_pred_every == 0 and cnt != 0:
                valid(valid_dataloader=valid_dataloader, model=model, epoch=epoch_cnt)
                print('updating saved snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.diffusion.state_dict(),
                            'optimizer_diff': optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir, 'flowdiff.pth'))

            if actual_step >= args.final_step:
                break

            cnt += 1

        scheduler.step()
        epoch_cnt += 1
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer_diff.param_groups[0]["lr"]))

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'diffusion': model.diffusion.state_dict(),
                'optimizer_diff': optimizer_diff.state_dict()},
               osp.join(args.snapshot_dir,
                        'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))
    end = timeit.default_timer()
    print(end - start, 'seconds')
