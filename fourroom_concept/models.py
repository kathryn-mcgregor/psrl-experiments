"""
Tortoise ORM models for the fourroom_concept experiment.

Two tables:
  - Step    — one row per action taken by a participant
  - Session — one row per completed participant
"""

from tortoise import fields
from tortoise.models import Model


class Step(Model):
    id           = fields.IntField(pk=True)
    seed         = fields.CharField(max_length=64, index=True)
    timestamp    = fields.DatetimeField()
    maze_idx     = fields.IntField()
    maze_step    = fields.IntField()
    action       = fields.CharField(max_length=32)
    prev_pos     = fields.CharField(max_length=32)   # JSON "[row, col]"
    new_pos      = fields.CharField(max_length=32)   # JSON "[row, col]"
    moved        = fields.BooleanField()
    visited_goal = fields.IntField(null=True)
    left_goal    = fields.IntField(null=True)
    reward       = fields.IntField()

    class Meta:
        table = "steps"


class Session(Model):
    id           = fields.IntField(pk=True)
    seed         = fields.CharField(max_length=64, unique=True, index=True)
    completed_at = fields.DatetimeField()
    mode         = fields.CharField(max_length=32)
    dims         = fields.JSONField()                # list of dim names
    n_kinds      = fields.JSONField()                # {dim: int}
    n_goals      = fields.IntField()
    rule_dim     = fields.CharField(max_length=32)
    rule_value   = fields.CharField(max_length=32)
    total_score  = fields.IntField()
    total_steps  = fields.IntField()
    log          = fields.JSONField()                # list of goal-visit events
    mazes        = fields.JSONField()                # list of maze layout dicts

    class Meta:
        table = "sessions"
