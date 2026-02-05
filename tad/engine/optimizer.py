import torch


def build_optimizer(cfg, model, logger):

    # Compat for both DDP (model.module) and Single GPU (model)
    raw_model = model.module if hasattr(model, "module") else model

    # set the backbone's optim_groups: SHOULD ONLY CONTAIN BACKBONE PARAMS
    if hasattr(raw_model, "backbone"):  # if backbone exists
        if raw_model.backbone.freeze_backbone is False:  # not frozen
            assert (
                "backbone" in cfg.keys()
            ), "Freeze_backbone is set to False, but backbone parameters is not provided in the optimizer config."
            backbone_cfg = cfg["backbone"]
            cfg.pop("backbone")
            backbone_optim_groups = get_backbone_optim_groups(backbone_cfg, raw_model, logger)

        else:  # frozen backbone
            backbone_optim_groups = []
            logger.info("Freeze the backbone...")
    else:
        backbone_optim_groups = []

    # set the detector's optim_groups: SHOULD NOT CONTAIN BACKBONE PARAMS
    # here, if each method want their own paramwise config, eg. to specify the learning rate,
    # weight decay for a certain layer, the model should have a function called get_optim_groups
    paramwise = cfg.pop("paramwise", True)
    if paramwise:
        det_optim_groups = raw_model.get_optim_groups(cfg)
    else:
        # optim_groups that does not contain backbone params
        detector_params = []
        for name, param in raw_model.named_parameters():
            # exclude the backbone
            if name.startswith("backbone"):
                continue
            detector_params.append(param)
        det_optim_groups = [dict(params=detector_params)]

    # merge the optim_groups
    optim_groups = backbone_optim_groups + det_optim_groups

    optimizer = torch.optim.AdamW(optim_groups, **cfg)

    return optimizer


def _group_backbone_parameters(raw_model, exclude_name_list, custom_name_list):
    custom_params_list = [[] for _ in custom_name_list]
    rest_params_list = []
    name_list = []

    for name, param in raw_model.backbone.named_parameters():
        # Check exclusion
        is_exclude = any(exclude_name in name for exclude_name in exclude_name_list)
        if is_exclude:
            continue

        # Check custom groups
        is_custom = False
        for i, custom_name in enumerate(custom_name_list):
            if custom_name in name:
                custom_params_list[i].append(param)
                name_list.append(name)
                is_custom = True
                break

        if is_custom:
            continue

        # Rest parameters
        rest_params_list.append(param)
        name_list.append(name)

    return rest_params_list, custom_params_list, name_list


def get_backbone_optim_groups(cfg, model, logger):
    """Example:
    backbone = dict(
        lr=1e-5,
        weight_decay=1e-4,
        custom=[dict(name="residual", lr=1e-3, weight_decay=1e-4)],
        exclude=[],
    )
    """
    custom_cfg = cfg.get("custom", [])
    custom_name_list = [d["name"] for d in custom_cfg]
    exclude_name_list = cfg.get("exclude", [])

    # Compat for both DDP (model.module) and Single GPU (model)
    raw_model = model.module if hasattr(model, "module") else model

    rest_params_list, custom_params_list, name_list = _group_backbone_parameters(
        raw_model, exclude_name_list, custom_name_list
    )

    for name in name_list:
        logger.info(f"Backbone parameter: {name}")

    # add params to optim_groups
    backbone_optim_groups = []

    if len(rest_params_list) > 0:
        backbone_optim_groups.append(
            dict(
                params=rest_params_list,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )
        )

    for i, _ in enumerate(custom_name_list):
        if len(custom_params_list[i]) > 0:
            backbone_optim_groups.append(
                dict(
                    params=custom_params_list[i],
                    lr=custom_cfg[i]["lr"],
                    weight_decay=custom_cfg[i]["weight_decay"],
                )
            )
    return backbone_optim_groups
