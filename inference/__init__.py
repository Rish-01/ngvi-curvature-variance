from .bbvi_adam import BBVIAdam
from .natural_gradient import NaturalGradientVI
from .diagonal_fisher import DiagonalFisherVI
from .variance_reduced import VarianceReducedBBVI

_METHODS = {
    "bbvi_adam": BBVIAdam,
    "natural_gradient": NaturalGradientVI,
    "diagonal_fisher": DiagonalFisherVI,
    "variance_reduced": VarianceReducedBBVI,
}


def _coerce_method_kwargs(kwargs: dict) -> dict:
    """
    PyYAML's default (YAML 1.1) parses unquoted ``1e-3`` as a string, not a float.
    Normalize hyperparameters so torch and numeric code see real numbers.
    """
    out = dict(kwargs)
    for key in ("lr", "eps_adam", "damping", "fisher_ema"):
        if key in out and isinstance(out[key], str):
            out[key] = float(out[key])
    if "n_samples" in out and isinstance(out["n_samples"], str):
        out["n_samples"] = int(float(out["n_samples"]))
    if "betas" in out and isinstance(out["betas"], (list, tuple)):
        seq = out["betas"]
        out["betas"] = type(seq)(
            float(x) if isinstance(x, str) else x for x in seq
        )
    return out


def get_inference_method(name: str, *, model, D: int, **kwargs):
    """
    Construct an inference object from a config key name.

    Parameters
    ----------
    name : str
        One of ``bbvi_adam``, ``natural_gradient``, ``diagonal_fisher``,
        ``variance_reduced``.
    model, D
        Passed through to the method constructor.
    **kwargs
        Hyperparameters from the YAML ``methods.<name>`` block.
    """
    cls = _METHODS.get(name)
    if cls is None:
        known = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown inference method {name!r}. Expected one of: {known}.")

    kwargs = _coerce_method_kwargs(kwargs)

    if name in ("bbvi_adam", "variance_reduced"):
        kwargs = dict(kwargs)
        if "betas" in kwargs and isinstance(kwargs["betas"], list):
            kwargs["betas"] = tuple(kwargs["betas"])

    return cls(model=model, D=D, **kwargs)


__all__ = [
    "BBVIAdam",
    "NaturalGradientVI",
    "DiagonalFisherVI",
    "VarianceReducedBBVI",
    "get_inference_method",
]
