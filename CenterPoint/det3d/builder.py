import logging
import pickle


def build_dbsampler(cfg, logger=None):
    """Build CenterPoint database sampler lazily for training augmentation."""
    import det3d.core.sampler.preprocess as prep
    from det3d.core.sampler.preprocess import DataBasePreprocessor
    from det3d.core.sampler.sample_ops import DataBaseSamplerV2

    logger = logger or logging.getLogger("build_dbsampler")
    prepors = []
    for prep_cfg in cfg.db_prep_steps:
        if "filter_by_difficulty" in prep_cfg:
            prepors.append(prep.DBFilterByDifficulty(
                prep_cfg["filter_by_difficulty"],
                logger=logger,
            ))
        elif "filter_by_min_num_points" in prep_cfg:
            prepors.append(prep.DBFilterByMinNumPoint(
                prep_cfg["filter_by_min_num_points"],
                logger=logger,
            ))
        else:
            raise ValueError(f"Unknown database prep type: {prep_cfg}")

    with open(cfg.db_info_path, "rb") as f:
        db_infos = pickle.load(f)

    grot_range = list(cfg.global_random_rotation_range_per_object)
    if len(grot_range) == 0:
        grot_range = None

    return DataBaseSamplerV2(
        db_infos,
        cfg.sample_groups,
        DataBasePreprocessor(prepors),
        cfg.rate,
        grot_range,
        logger=logger,
    )


def _create_learning_rate_scheduler(optimizer, learning_rate_config, total_step):
    """Create the CenterPoint learning-rate scheduler used by train.py."""
    from det3d.solver import learning_schedules_fastai as lsf

    lr_scheduler = None
    learning_rate_type = learning_rate_config.type
    cfg = learning_rate_config

    if learning_rate_type == "multi_phase":
        lr_phases = []
        mom_phases = []
        for phase_cfg in cfg.phases:
            lr_phases.append((phase_cfg.start, phase_cfg.lambda_func))
            mom_phases.append((phase_cfg.start, phase_cfg.momentum_lambda_func))
        lr_scheduler = lsf.LRSchedulerStep(optimizer, total_step, lr_phases, mom_phases)
    elif learning_rate_type == "one_cycle":
        lr_scheduler = lsf.OneCycle(
            optimizer,
            total_step,
            cfg.lr_max,
            cfg.moms,
            cfg.div_factor,
            cfg.pct_start,
        )
    elif learning_rate_type == "exponential_decay":
        lr_scheduler = lsf.ExponentialDecay(
            optimizer,
            cfg.initial_learning_rate,
            cfg.decay_length,
            cfg.decay_factor,
            cfg.staircase,
        )
    elif learning_rate_type == "manual_stepping":
        lr_scheduler = lsf.ManualStepping(
            optimizer,
            total_step,
            cfg.boundaries,
            cfg.rates,
        )

    if lr_scheduler is None:
        raise ValueError(f"Learning rate {learning_rate_type} not supported")
    return lr_scheduler
