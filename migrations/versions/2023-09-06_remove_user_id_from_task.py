"""remove user_id from task

Revision ID: ec0e168fe34a
Revises: a9b338000620
Create Date: 2023-09-06 01:13:32.868983

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ec0e168fe34a'
down_revision: Union[str, None] = 'a9b338000620'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('task_user_id_fkey', 'task', type_='foreignkey')
    op.drop_column('task', 'user_id')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('task', sa.Column('user_id', sa.UUID(), autoincrement=False, nullable=False))
    op.create_foreign_key('task_user_id_fkey', 'task', 'user', ['user_id'], ['id'])
    # ### end Alembic commands ###