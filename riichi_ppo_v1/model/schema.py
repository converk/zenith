"""Versioned serialization constants shared by PPO and SFT."""

TOKEN_SCHEMA_VERSION = 13

# v13 action-query rows use the existing candidate segment.  Keeping these
# values in one module prevents the encoder and attention layout from silently
# disagreeing about which rows are policy queries.
from .feature_schema import (
    ACTION_QUERY_DEFENSE,
    ACTION_QUERY_OFFENSE,
    ACTION_QUERY_SEGMENT,
)
