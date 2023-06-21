"""create table user farm prediction sheep

Revision ID: 66c9022e0947
Revises: 
Create Date: 2023-06-21 04:27:50.223099

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = '66c9022e0947'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('farm',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('location', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_farm_name'), 'farm', ['name'], unique=False)
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('email', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('password', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('sheep',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('tag', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('farm_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['farm_id'], ['farm.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sheep_tag'), 'sheep', ['tag'], unique=False)
    op.create_table('prediction',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('predicted_weight', sa.Float(), nullable=False),
    sa.Column('actual_weight', sa.Float(), nullable=False),
    sa.Column('sheep_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['sheep_id'], ['sheep.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_prediction_actual_weight'), 'prediction', ['actual_weight'], unique=False)
    op.create_index(op.f('ix_prediction_predicted_weight'), 'prediction', ['predicted_weight'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_prediction_predicted_weight'), table_name='prediction')
    op.drop_index(op.f('ix_prediction_actual_weight'), table_name='prediction')
    op.drop_table('prediction')
    op.drop_index(op.f('ix_sheep_tag'), table_name='sheep')
    op.drop_table('sheep')
    op.drop_table('user')
    op.drop_index(op.f('ix_farm_name'), table_name='farm')
    op.drop_table('farm')
    # ### end Alembic commands ###
