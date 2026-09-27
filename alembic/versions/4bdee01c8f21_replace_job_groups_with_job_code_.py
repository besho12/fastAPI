"""replace job groups with job code grouping

Revision ID: 4bdee01c8f21
Revises: bd61e6cc9641
Create Date: 2026-09-01 22:14:46.599623

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "4bdee01c8f21"
down_revision: Union[str, Sequence[str], None] = "bd61e6cc9641"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Replace the old Job Group based schema with Job Code based grouping.

    New design:
        job_code    -> grouping key + job identity
        company_code -> company identifier

    Removed:
        job_group_id
        company_name
        job_groups table
    """

    # ---------------------------------------------------------
    # 1. Drop old unique constraints
    # ---------------------------------------------------------
    op.drop_constraint(
        "uq_jobs_content_hash",
        "jobs",
        type_="unique",
    )

    op.drop_constraint(
        "uq_jobs_raw_content_hash",
        "jobs",
        type_="unique",
    )

    # ---------------------------------------------------------
    # 2. Drop foreign key from jobs.job_group_id
    # ---------------------------------------------------------
    op.drop_constraint(
        "jobs_job_group_id_fkey",
        "jobs",
        type_="foreignkey",
    )

    # ---------------------------------------------------------
    # 3. Drop old indexes
    # ---------------------------------------------------------
    op.drop_index(
        "ix_jobs_job_group_id",
        table_name="jobs",
    )

    op.drop_index(
        "ix_jobs_company_name",
        table_name="jobs",
    )

    # ---------------------------------------------------------
    # 4. Remove old Job Group column
    # ---------------------------------------------------------
    op.drop_column(
        "jobs",
        "job_group_id",
    )

    # ---------------------------------------------------------
    # 5. Remove old company_name column
    # ---------------------------------------------------------
    op.drop_column(
        "jobs",
        "company_name",
    )

    # ---------------------------------------------------------
    # 6. job_code is now required
    # ---------------------------------------------------------
    op.alter_column(
        "jobs",
        "job_code",
        existing_type=sa.String(length=64),
        nullable=False,
    )

    # ---------------------------------------------------------
    # 7. company_code is now required
    # ---------------------------------------------------------
    op.alter_column(
        "jobs",
        "company_code",
        existing_type=sa.String(length=64),
        nullable=False,
    )

    # ---------------------------------------------------------
    # 8. New uniqueness rules
    #
    # Same job_code can have multiple versions.
    # But the exact same content under the same job_code
    # must not be inserted twice.
    # ---------------------------------------------------------
    op.create_unique_constraint(
        "uq_jobs_job_code_content_hash",
        "jobs",
        ["job_code", "content_hash"],
    )

    op.create_unique_constraint(
        "uq_jobs_job_code_raw_content_hash",
        "jobs",
        ["job_code", "raw_content_hash"],
    )

    # ---------------------------------------------------------
    # 9. Remove the old Job Groups table
    # ---------------------------------------------------------
    op.drop_table("job_groups")


def downgrade() -> None:
    """
    Restore the old Job Group based schema.
    """

    # ---------------------------------------------------------
    # 1. Re-create job_groups table
    # ---------------------------------------------------------
    op.create_table(
        "job_groups",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "normalized_name",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "company_code",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "normalized_name",
            name="uq_job_groups_normalized_name",
        ),
    )

    # ---------------------------------------------------------
    # 2. Restore company_name
    # ---------------------------------------------------------
    op.add_column(
        "jobs",
        sa.Column(
            "company_name",
            sa.String(length=255),
            nullable=True,
        ),
    )

    # ---------------------------------------------------------
    # 3. Restore job_group_id
    # ---------------------------------------------------------
    op.add_column(
        "jobs",
        sa.Column(
            "job_group_id",
            sa.Integer(),
            nullable=True,
        ),
    )

    # ---------------------------------------------------------
    # 4. Remove new unique constraints
    # ---------------------------------------------------------
    op.drop_constraint(
        "uq_jobs_job_code_content_hash",
        "jobs",
        type_="unique",
    )

    op.drop_constraint(
        "uq_jobs_job_code_raw_content_hash",
        "jobs",
        type_="unique",
    )

    # ---------------------------------------------------------
    # 5. Restore old unique constraints
    # ---------------------------------------------------------
    op.create_unique_constraint(
        "uq_jobs_content_hash",
        "jobs",
        ["content_hash"],
    )

    op.create_unique_constraint(
        "uq_jobs_raw_content_hash",
        "jobs",
        ["raw_content_hash"],
    )

    # ---------------------------------------------------------
    # 6. Restore FK
    # ---------------------------------------------------------
    op.create_foreign_key(
        "jobs_job_group_id_fkey",
        "jobs",
        "job_groups",
        ["job_group_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # ---------------------------------------------------------
    # 7. Restore indexes
    # ---------------------------------------------------------
    op.create_index(
        "ix_jobs_job_group_id",
        "jobs",
        ["job_group_id"],
    )

    op.create_index(
        "ix_jobs_company_name",
        "jobs",
        ["company_name"],
    )

    # ---------------------------------------------------------
    # 8. job_code/company_code become nullable again
    # ---------------------------------------------------------
    op.alter_column(
        "jobs",
        "job_code",
        existing_type=sa.String(length=64),
        nullable=True,
    )

    op.alter_column(
        "jobs",
        "company_code",
        existing_type=sa.String(length=64),
        nullable=True,
    )