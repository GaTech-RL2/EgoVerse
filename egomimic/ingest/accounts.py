"""Dashboard accounts, kept in the bucket (pipeline/accounts.json).

    python -m egomimic.ingest.accounts add <user> --vendor <slug> | --staff
    python -m egomimic.ingest.accounts password <user>
    python -m egomimic.ingest.accounts remove <user>
    python -m egomimic.ingest.accounts list

A vendor account sees only its vendor's deliveries; a staff account sees every vendor
and (later) approves. We create the accounts: there is no sign-up. Passwords are stored
as salted scrypt hashes, never in the clear. Store flags as for egomimic.ingest.watch.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
import sys

from egomimic.ingest.store import VENDOR, LocalStore, R2Store, Store, now

ROLES = ("vendor", "staff")
MIN_PASSWORD = 12
# scrypt cost: ~50 ms and 16 MiB per check, so a stolen accounts file is slow to crack.
SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def hash_password(password: str, salt: bytes | None = None) -> dict:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    return {"scrypt": {**SCRYPT, "salt": salt.hex(), "hash": digest.hex()}}


def check_password(account: dict | None, password: str) -> bool:
    """Constant-time; an unknown user costs the same as a wrong password."""
    rec = (account or {}).get("password", {}).get("scrypt")
    if rec is None:
        hash_password(password)  # same work as a real check
        return False
    params = {k: rec[k] for k in ("n", "r", "p")}
    digest = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(rec["salt"]), dklen=32, **params
    )
    return hmac.compare_digest(digest.hex(), rec["hash"])


def add(
    store: Store, user: str, password: str, *, role: str, vendor: str | None = None
) -> dict:
    if not (VENDOR.fullmatch(user) and len(user) <= 64):  # also the reviewer name
        raise ValueError(f"{user!r}: lowercase letters and digits, joined by _ or -")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if role == "vendor" and not (vendor and VENDOR.fullmatch(vendor)):
        raise ValueError("a vendor account needs a vendor slug")
    if role == "staff" and vendor:
        raise ValueError("a staff account has no vendor")
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"passwords need at least {MIN_PASSWORD} characters")
    accounts = store.get_accounts()
    if user in accounts:
        raise ValueError(f"{user} already exists")
    accounts[user] = {
        "role": role,
        "vendor": vendor if role == "vendor" else None,
        "password": hash_password(password),
        "created": now(),
    }
    store.put_accounts(accounts)
    return accounts[user]


def set_password(store: Store, user: str, password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"passwords need at least {MIN_PASSWORD} characters")
    accounts = store.get_accounts()
    if user not in accounts:
        raise KeyError(user)
    accounts[user]["password"] = hash_password(password)
    accounts[user]["password_set"] = now()
    store.put_accounts(accounts)


def remove(store: Store, user: str) -> None:
    accounts = store.get_accounts()
    if accounts.pop(user, None) is None:
        raise KeyError(user)
    store.put_accounts(accounts)


def store_from_args(args) -> Store:
    if args.local_store:
        return LocalStore(args.local_store)
    return R2Store(args.bucket, root=args.root)


def add_store_args(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bucket", default="rldb", help="R2 bucket (default rldb)")
    src.add_argument("--local-store", help="a directory instead of R2")
    ap.add_argument("--root", default="", help="R2 key prefix, e.g. _test/run1/")


def _new_password(user: str) -> str:
    first = getpass.getpass(f"password for {user}: ")
    if first != getpass.getpass("again: "):
        raise SystemExit("passwords differ")
    return first


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_store_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("user")
    who = a.add_mutually_exclusive_group(required=True)
    who.add_argument("--vendor")
    who.add_argument("--staff", action="store_true")
    sub.add_parser("password").add_argument("user")
    sub.add_parser("remove").add_argument("user")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    store = store_from_args(args)
    try:
        if args.cmd == "add":
            role = "staff" if args.staff else "vendor"
            add(
                store,
                args.user,
                _new_password(args.user),
                role=role,
                vendor=args.vendor,
            )
        elif args.cmd == "password":
            set_password(store, args.user, _new_password(args.user))
        elif args.cmd == "remove":
            remove(store, args.user)
        else:
            listing = {
                u: {k: a.get(k) for k in ("role", "vendor", "created")}
                for u, a in sorted(store.get_accounts().items())
            }
            print(json.dumps(listing, indent=2))
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
