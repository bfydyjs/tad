from tad.utils.registry import Registry

EVALUATORS = Registry("evaluator")


def build_evaluator(cfg):
    """Build evaluator."""
    return EVALUATORS.build(cfg)
