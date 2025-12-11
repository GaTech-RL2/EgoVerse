HOST="lowuse-pg.cxkes66cc93x.us-east-1.rds.amazonaws.com"
DBUSER=POSTGRES_ADMIN
DBPASS='STRONG_PASSWORD_HERE'

psql "host=$HOST port=5432 user=$DBUSER password=$DBPASS dbname=postgres sslmode=require" <<'SQL'
CREATE DATABASE appdb;
CREATE USER appuser WITH PASSWORD 'APPUSER_STRONG_PW';
GRANT CONNECT ON DATABASE appdb TO appuser;
\c appdb
CREATE SCHEMA app AUTHORIZATION appuser;
GRANT ALL PRIVILEGES ON SCHEMA app TO appuser;
SQL