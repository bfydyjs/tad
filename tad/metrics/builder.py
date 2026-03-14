from tad.utils.registry import Registry

EVALUATORS = Registry("evaluators")


def build_evaluator(cfg):
    """Build evaluator."""
    return EVALUATORS.build(cfg)
