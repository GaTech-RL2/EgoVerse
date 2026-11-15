"""Verified, compressed backups for compacting the local dashboard cache.

All experts, trace versions and trace links are retained. Only superseded full
dashboard snapshots are expired; each snapshot includes the full loss histories.
Neither backup nor restore removes or overwrites an existing file.
"""
from contextlib import closing
import hashlib
import json
import lzma
import os
from pathlib import Path
import sqlite3

from .server import connect, pack_payload, unpack_payload, SNAPSHOT_RETENTION

TABLES = {'snapshots': ('id', 3, 1), 'experts': ('oracle_sha,state_id', 4, 3),
          'trace_versions': ('id', 2, 1), 'traces': ('run,state_id,seed', None, None)}


def line(value):
    return (json.dumps(value, separators=(',', ':'), ensure_ascii=True)+'\n').encode()


def normalized(table, row):
    values=list(row); _, payload, digest=TABLES[table]
    if payload is not None:
        values[payload]=unpack_payload(values[payload])
        if hashlib.sha256(values[payload].encode()).hexdigest()!=values[digest]:
            raise ValueError('Payload checksum mismatch: '+table)
    return values


def backup(source, archive, keep=SNAPSHOT_RETENTION, progress=None):
    if keep<1:raise ValueError('At least the latest snapshot must be retained')
    source=Path(source);archive=Path(archive)
    counts={};digest=hashlib.sha256()
    with closing(sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN')
        names={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if names!=set(TABLES):raise ValueError('Unexpected database tables; review before compacting')
        header={'format':'leact-tracker-backup-v1','keep_snapshots':keep,
                'source':str(source.resolve()),'source_bytes':source.stat().st_size,
                'source_counts':{t:db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in TABLES}}
        with archive.open('xb') as output:
            with lzma.LZMAFile(output,'wb',preset=3) as compressed:
                compressed.write(line({'header':header}))
                for table,(order,_,_) in TABLES.items():
                    query='SELECT * FROM '+table
                    params=()
                    if table=='snapshots':
                        query+=' WHERE id IN (SELECT id FROM snapshots ORDER BY id DESC LIMIT ?)';params=(keep,)
                    counts[table]=0
                    for row in db.execute(query+' ORDER BY '+order,params):
                        raw=line([table,normalized(table,row)])
                        digest.update(raw);compressed.write(raw);counts[table]+=1
                        if progress and counts[table]%10000==0:
                            progress({'table':table,'rows':counts[table],'archive_bytes':output.tell()})
                manifest={'counts':counts,'sha256':digest.hexdigest()}
                compressed.write(line({'manifest':manifest}))
            output.flush();os.fsync(output.fileno())
    return {**manifest,'archive_bytes':archive.stat().st_size,'source':header}


def records(archive):
    counts={t:0 for t in TABLES};digest=hashlib.sha256();manifest=None
    with lzma.open(archive,'rb') as stream:
        header=json.loads(next(stream))['header']
        if header['format']!='leact-tracker-backup-v1':raise ValueError('Unknown backup format')
        for raw in stream:
            value=json.loads(raw)
            if isinstance(value,dict):
                manifest=value['manifest']
                if stream.read(1):raise ValueError('Unexpected trailing backup data')
                break
            table,row=value
            normalized(table,row)
            digest.update(raw);counts[table]+=1
            yield table,row
    if manifest!={'counts':counts,'sha256':digest.hexdigest()}:
        raise ValueError('Backup is incomplete or its content checksum changed')
    expected=dict(header['source_counts'])
    expected['snapshots']=min(expected['snapshots'],header['keep_snapshots'])
    if counts!=expected:raise ValueError('Backup lost records')


def verify(archive):
    counts={t:0 for t in TABLES};digest=hashlib.sha256()
    for table,row in records(archive):
        counts[table]+=1;digest.update(line([table,row]))
    return {'counts':counts,'sha256':digest.hexdigest()}


def restore(archive, destination, progress=None):
    destination=Path(destination)
    # Refuse to touch any existing database, even an empty one.
    with destination.open('xb'):pass
    counts={t:0 for t in TABLES};digest=hashlib.sha256()
    with closing(connect(destination)) as db:
        for table,row in records(archive):
            digest.update(line([table,row]));counts[table]+=1
            _,payload,_=TABLES[table]
            if payload is not None:row[payload]=pack_payload(row[payload])
            db.execute('INSERT INTO '+table+' VALUES('+','.join('?' for _ in row)+')',row)
            if sum(counts.values())%256==0:db.commit()
            if progress and counts[table]%10000==0:progress({'table':table,'rows':counts[table]})
        db.commit()
        check=hashlib.sha256();actual={}
        for table,(order,_,_) in TABLES.items():
            actual[table]=0
            for row in db.execute('SELECT * FROM '+table+' ORDER BY '+order):
                check.update(line([table,normalized(table,row)]));actual[table]+=1
        if actual!=counts or check.digest()!=digest.digest():raise ValueError('Restored database differs from backup')
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Restored database failed integrity check')
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    return {'counts':counts,'sha256':digest.hexdigest(),'database_bytes':destination.stat().st_size}
