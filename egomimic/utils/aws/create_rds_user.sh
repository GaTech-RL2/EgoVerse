# ---------- config ----------
HOST="lowuse-pg.cxkes66cc93x.us-east-1.rds.amazonaws.com"

MASTER_USER="POSTGRES_ADMIN"
MASTER_PASS="STRONG_PASSWORD_HERE"

APP_DB="appdb"
APP_USER="appuser"
APP_PASS="APPUSER_PASSWORD_HERE"

# ---------- 1) allow appuser to create a schema in appdb (run as master) ----------
PGPASSWORD="$MASTER_PASS" psql "host=$HOST port=5432 user=$MASTER_USER dbname=$APP_DB sslmode=require" -v ON_ERROR_STOP=1 <<'SQL'
GRANT CREATE ON DATABASE appdb TO appuser;
SQL
