from enum import Enum


class Tier(str, Enum):
    CACHE = "CACHE"
    SSD = "SSD"
    HDD = "HDD"
    ARCHIVE = "ARCHIVE"