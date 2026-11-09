"""Explicit scientific-version dispatch; the historical v1 contract is untouched."""
from reid.utils import lodo_protocol as v1
from reid.utils.lodo_protocol import has_lodo_protocol, require_explicit_lodo


def backend(cfg):
    from reid.utils import mgn_lodo_protocol as v2
    return v2 if v2.is_mgn_lodo(cfg) else v1


def validate_lodo_config(cfg, *, project_root=None):
    return backend(cfg).validate_lodo_config(cfg,project_root=project_root)


def validate_output_path(cfg, *, project_root=None):
    return backend(cfg).validate_output_path(cfg,project_root=project_root)


def scientific_configuration(cfg):
    return backend(cfg).scientific_configuration(cfg)


def configuration_identity(cfg):
    return backend(cfg).configuration_identity(cfg)
