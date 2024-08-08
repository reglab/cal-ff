# California CAFO

## Env

You need a .env file in your working directory that should look something like this:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=55550
POSTGRES_DB=cacafo
POSTGRES_USER={fill in user}
POSTGRES_PASSWORD={fill in password}
DATA_ROOT={path to data}
SSH_ALIAS=lcr
SSH_LOCAL_PORT=55550
SSH_REMOTE_PORT=5432
```

And then in your `.ssh/config` file you should have:
```
Host lcr
	HostName lc-r-2.law.stanford.edu
	User {fill in user}
	Port 5988
```
