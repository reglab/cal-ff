# Cal-FF: A Comprehensive Dataset of Factory Farms in California Compiled Using Computer Vision and Human Validation

This repository contains code necessary to post-process annotations to generate final facility outputs for the Cal-FF dataset, described below.

It does not contain the minimal custom code used to train the YOLO model used to stratify images for human labeling. No data directly from this model appears in the final dataset.

## Basic Usage

The `cacafo` package can be installed with any Python package manager; with `pip`, this would be as simple as 1) creating a venv and 2) running `pip install -e .` from the project directory.

Once the project is installed, you should have two executables: 1) the `calf` executable, which includes a number of scripts to make working with the project a bit easier, and 2) the `alembic` executable which is used to make and run database migrations.

## Environment Setup

You need a .env file in your working directory that should look something like this:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=55550
POSTGRES_DB=cacafo
POSTGRES_USER={fill in user}
POSTGRES_PASSWORD={fill in password}
DATA_ROOT={path to data}
# if you want to manage your tunnel with calf shell commands
SSH_ALIAS=lcr
SSH_LOCAL_PORT=55550
SSH_REMOTE_PORT=5432
```

And then in your `.ssh/config` file you should have, if you're using the database on lcr:
```
Host lcr
	HostName lc-r-2.law.stanford.edu
	User {fill in user}
	Port 5988
```

## Database

Before running migrations/creating the db, you will have to manually
run `CREATE EXTENSION postgis;` as a superuser on the db.
