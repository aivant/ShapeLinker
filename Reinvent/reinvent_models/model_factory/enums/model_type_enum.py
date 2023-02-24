from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTypeEnum:
    DEFAULT = "default"
    REINVENT_CORE = "reinvent_core"
    LIB_INVENT = "lib_invent"
    LINK_INVENT = "link_invent"