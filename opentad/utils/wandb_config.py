def get_custom_config(cfg):
    # 1. return full config
    # return cfg

    # 2. return optional selected fields
    return {
        # "model": {
        #     "type": cfg.model.type,
        #     "projection": {
        #         "type": cfg.model.projection.type,
        #         "in_channels": cfg.model.projection.in_channels,
        #         "out_channels": cfg.model.projection.out_channels,
        #         "arch": cfg.model.projection.arch,
                "use_abs_pe": cfg.model.projection.use_abs_pe,
        #         "max_seq_len": cfg.model.projection.max_seq_len,
                "mlp_dim": cfg.model.projection.mlp_dim,
                "encoder_win_size": cfg.model.projection.encoder_win_size,
                "k": cfg.model.projection.k,
                "init_conv_vars": cfg.model.projection.init_conv_vars,
                "path_pdrop": cfg.model.projection.path_pdrop,
                "input_noise": cfg.model.projection.input_noise,
        #     },
        #     "neck": {
        #         "type": cfg.model.neck.type,
        #         "in_channels": cfg.model.neck.in_channels,
        #         "out_channels": cfg.model.neck.out_channels,
        #         "num_levels": cfg.model.neck.num_levels,
        #     },
        #     "rpn_head": {
        #         "type": cfg.model.rpn_head.type,
        #         "num_classes": cfg.model.rpn_head.num_classes,
        #         "in_channels": cfg.model.rpn_head.in_channels,
        #         "feat_channels": cfg.model.rpn_head.feat_channels,
        #         "num_convs": cfg.model.rpn_head.num_convs,
                "iou_loss_weight": cfg.model.rpn_head.iou_loss_weight,
        #         "cls_prior_prob": cfg.model.rpn_head.cls_prior_prob,
        #         "prior_generator": {
        #             "type": cfg.model.rpn_head.prior_generator.type,
        #             "strides": cfg.model.rpn_head.prior_generator.strides,
        #             "regression_range": cfg.model.rpn_head.prior_generator.regression_range,
        #         },
        #         "loss_normalizer": cfg.model.rpn_head.loss_normalizer,
        #         "loss_normalizer_momentum": cfg.model.rpn_head.loss_normalizer_momentum,
        #         "center_sample": cfg.model.rpn_head.center_sample,
        #         "center_sample_radius": cfg.model.rpn_head.center_sample_radius,
                "label_smoothing": cfg.model.rpn_head.label_smoothing,
        #         "loss": {
        #             "cls_loss": {
        #                 "type": cfg.model.rpn_head.loss.cls_loss.type,
        #             },
        #             "reg_loss": {
        #                 "type": cfg.model.rpn_head.loss.reg_loss.type,

        #             },
        #         },
        #     },
        # },
        # "solver": {
        #     "train": {
                "batch_size": cfg.solver.train.batch_size,
        #         "num_workers": cfg.solver.train.num_workers,
        #     },
        #     "val": {
        #         "batch_size": cfg.solver.val.batch_size,
        #         "num_workers": cfg.solver.val.num_workers,
        #     },
        #     "test": {
        #         "batch_size": cfg.solver.test.batch_size,
        #         "num_workers": cfg.solver.test.num_workers,
        #     },
        #     "clip_grad_norm": cfg.solver.clip_grad_norm,
        # },
        # "optimizer": {
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
            "paramwise": cfg.optimizer.paramwise,
        # },
        # "scheduler": {
            "warmup_epoch": cfg.scheduler.warmup_epoch,
            "max_epoch": cfg.scheduler.max_epoch,
        # },
        # "inference": {
        #     "load_from_raw_predictions": cfg.inference.load_from_raw_predictions,
        #     "save_raw_prediction": cfg.inference.save_raw_prediction,
        # },
        # "post_processing": {
        #     "nms":{
        #         "use_soft_nms": cfg.post_processing.nms.use_soft_nms,
        #         "sigma": cfg.post_processing.nms.sigma,
        #         "max_seg_num": cfg.post_processing.nms.max_seg_num,
        #         "min_score": cfg.post_processing.nms.min_score,
        #         "multiclass": cfg.post_processing.nms.multiclass,
        #         "voting_thresh": cfg.post_processing.nms.voting_thresh,
        #     },
        #     "voting_thresh": cfg.post_processing.nms.voting_thresh
        #     },
        # "workflow": {
        #     "logging_interval": cfg.workflow.logging_interval,
        #     "checkpoint_interval": cfg.workflow.checkpoint_interval,
        #     "val_eval_interval": cfg.workflow.val_eval_interval,
            "val_start_epoch": cfg.workflow.val_start_epoch,
        # },
    }