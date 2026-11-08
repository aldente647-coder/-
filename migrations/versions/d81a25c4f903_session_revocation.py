"""Revoke browser sessions after password changes and account toggles."""
from alembic import op
import sqlalchemy as sa

revision = "d81a25c4f903"
down_revision = "c4a1f27de8b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_user", sa.Column("session_version", sa.Integer(),
                  nullable=False, server_default="1"), schema="ops")


def downgrade() -> None:
    op.drop_column("app_user", "session_version", schema="ops")
