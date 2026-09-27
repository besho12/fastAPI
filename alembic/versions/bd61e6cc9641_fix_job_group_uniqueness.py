"""fix job_groups uniqueness: normalized_name only

Revision ID: bd61e6cc9641
Revises: 51c00480b6f8
Create Date: 2026-08-30

WHY THIS MIGRATION EXISTS
--------------------------
The original `job_groups` schema (revision 51c00480b6f8, "add job
groups") made `(normalized_name, company_code)` jointly unique. In
practice this meant a user selecting the SAME `job_group` (e.g. "Data
Scientist") for jobs from two different companies ended up with TWO
separate JobGroup rows -- which is wrong: the user-selected Job Group
name is supposed to be the sole authoritative comparison scope,
independent of company.

This migration:
  1. For every set of `job_groups` rows that share the same
     `normalized_name` (i.e. were incorrectly split by `company_code`),
     picks the OLDEST row (lowest `id`) as the canonical survivor.
  2. Repoints every `jobs.job_group_id` that referenced one of the
     non-canonical duplicates onto the canonical group's id, so no Job
     silently loses its group membership or its comparison history.
  3. Deletes the now-orphaned duplicate `job_groups` rows.
  4. Drops the old two-column unique constraint and replaces it with a
     single-column unique constraint on `normalized_name` alone,
     matching the corrected `app.database.models.JobGroup` definition.

`company_code` itself is NOT dropped from the table -- it remains as
informational metadata (see `JobGroup`'s docstring in
app/database/models.py) on whichever row survives the merge.

BACKWARD-SAFETY NOTES
----------------------
* No table is dropped. No `jobs` row is ever deleted.
* Every `jobs.job_group_id` that pointed at a merged-away group is
  UPDATEd to point at the survivor BEFORE that duplicate row is
  deleted, so no Job's group membership (and therefore no comparison
  scope going forward) is lost.
* If your `job_groups` table has no such duplicates yet, steps 1-3 are
  no-ops and only the constraint swap (step 4) applies.
* Downgrade restores the OLD constraint shape but cannot un-merge
  groups that were combined by the upgrade (that information -- which
  job originally belonged to which now-deleted duplicate group -- is
  not recoverable once step 3 has run). This is called out explicitly
  in `downgrade()` below.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "bd61e6cc9641"
down_revision: Union[str, Sequence[str], None] = "51c00480b6f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # --- 1 & 2 & 3: merge any company-split duplicate groups ---------
    # Repoint jobs.job_group_id from every non-canonical duplicate onto
    # the canonical (lowest-id) group for that normalized_name.
    op.execute(
        """
        WITH canonical AS (
            SELECT normalized_name, MIN(id) AS keep_id
            FROM job_groups
            GROUP BY normalized_name
        )
        UPDATE jobs
        SET job_group_id = canonical.keep_id
        FROM job_groups jg
        JOIN canonical ON canonical.normalized_name = jg.normalized_name
        WHERE jobs.job_group_id = jg.id
          AND jg.id <> canonical.keep_id;
        """
    )

    # Now that no `jobs` row references a non-canonical duplicate, it's
    # safe to delete the duplicates themselves.
    op.execute(
        """
        WITH canonical AS (
            SELECT normalized_name, MIN(id) AS keep_id
            FROM job_groups
            GROUP BY normalized_name
        )
        DELETE FROM job_groups jg
        USING canonical
        WHERE jg.normalized_name = canonical.normalized_name
          AND jg.id <> canonical.keep_id;
        """
    )

    # --- 4: re-key uniqueness on normalized_name alone ---------------
    op.drop_constraint(
        "uq_job_groups_normalized_name_company_code",
        "job_groups",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_job_groups_normalized_name",
        "job_groups",
        ["normalized_name"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Restores the OLD two-column constraint shape only. This CANNOT
    # restore duplicate groups that the upgrade merged (that data is
    # gone once step 3 above deleted the duplicate rows) -- downgrading
    # after duplicates were actually merged in production will simply
    # leave you with fewer, correctly-merged groups under the old
    # constraint shape, not a true reversal.
    op.drop_constraint(
        "uq_job_groups_normalized_name",
        "job_groups",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_job_groups_normalized_name_company_code",
        "job_groups",
        ["normalized_name", "company_code"],
    )