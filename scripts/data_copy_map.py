import sys
import datajoint as dj

from datajoint_utilities.dj_data_copy import db_migration


# Set up connection credentials and store config
source_conn = dj.connection.Connection(
    dj.config["database.host"],
    dj.config["database.user"],
    dj.config["database.password"],
)

target_conn = dj.connection.Connection(
    dj.config["custom"]["target_database.host"],
    dj.config["custom"]["target_database.user"],
    dj.config["custom"]["target_database.password"],
)

# Set up schema/table name mapping

source_db_prefix = "map_v2_"
target_db_prefix = dj.config["custom"]["target_database.db_prefix"]


schema_name_mapper = {
    source_db_prefix + schema_name: target_db_prefix + schema_name
    for schema_name in (
        "lab",
        "ccf",
        "experiment",
        "ephys",
        "tracking",
        "histology",
        # "psth",
        # "report",
    )
}

table_block_list = {
    f"{target_db_prefix}lab": [],
    f"{target_db_prefix}ccf": [],
    f"{target_db_prefix}experiment": [],
    f"{target_db_prefix}ephys": ["ArchivedClustering", "UnitPassingCriteria"],
    f"{target_db_prefix}tracking": [],
    f"{target_db_prefix}histology": ["ArchivedElectrodeHistology"],
    f"{target_db_prefix}psth": [],
}


def copy_data(restriction, batch_size=10, force_fetch=True):
    for source_schema_name, target_schema_name in schema_name_mapper.items():
        source_schema = dj.create_virtual_module(
            source_schema_name, source_schema_name, connection=source_conn
        )
        target_schema = dj.create_virtual_module(
            target_schema_name, target_schema_name, connection=target_conn
        )

        # update target external stores
        for store, spec in dj.config.get("stores", {}).items():
            target_spec = dj.config["custom"].get("target_stores", {}).get(store, spec)
            target_schema.schema.external[store].spec = {
                **target_schema.schema.external[store].spec,
                **target_spec,
            }

        db_migration.migrate_schema(
            source_schema,
            target_schema,
            restriction=restriction,
            table_block_list=table_block_list.get(target_schema_name, []),
            allow_missing_destination_tables=True,
            force_fetch=force_fetch,
            batch_size=batch_size,
        )


def main(offset=None, limit=None, session_batch_size=2):
    experiment = dj.create_virtual_module(
        f"{source_db_prefix}experiment",
        f"{source_db_prefix}experiment",
        connection=source_conn,
    )

    project_name = 'Brain-wide neural activity underlying memory-guided movement'
    session_query = (experiment.Session & (experiment.ProjectSession
                                           & {'project_name': project_name}))

    # Session restriction
    session_restriction = session_query.fetch("KEY", offset=offset, limit=limit)

    total_count = len(session_restriction)
    for i in range(0, total_count, session_batch_size):
        print(f"\n\t\t\t--------- PROCESSING {i}/{total_count} ---------\n")
        copy_data(session_restriction[i: i + session_batch_size])


if __name__ == "__main__":
    main(*sys.argv[1:])