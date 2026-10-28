#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging 
import time
from typing import Dict, List, Tuple, Sequence, Optional, Any
from pyrfc import Connection, CommunicationError, LogonError, ABAPApplicationError, ABAPRuntimeError
from log_utils import setup_logger

log = setup_logger("SAP_RTAB", "sap_rtab.log")

MAX_OPT = 72

def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]

def split_where(where: str, max_len: int = MAX_OPT) -> List[str]:
    parts: List[str] = []
    s = where.strip()
    while len(s) > max_len:
        cut_and = s.rfind(" AND ", 0, max_len)
        cut_or = s.rfind(" OR ", 0, max_len)
        cut = max(cut_and, cut_or)
        if cut <= 0:
            parts.append(s[:max_len])
            s = s[max_len:]
        else:
            parts.append(s[:cut].strip())
            s = s[cut + 1 :].strip()
    if s:
        parts.append(s)
    return [p for p in parts if p]

def options_from_where(where: str) -> List[Dict[str, str]]:
    where = (where or "").strip()
    if not where:
        return []
    return [{"TEXT": ln} for ln in split_where(where)]

def rfc_read_table(
    conn: Connection,
    table: str,
    fields: Sequence[str],
    where: str = "",
    rowcount: int = 0,
    rowskips: int = 0,
    delimiter: str = "§",
) -> List[Dict[str, str]]:

    try:
        res = conn.call(
            "RFC_READ_TABLE",
            QUERY_TABLE=table,
            DELIMITER=delimiter,
            FIELDS=[{"FIELDNAME": f} for f in fields],
            OPTIONS=options_from_where(where),
            ROWCOUNT=rowcount,
            ROWSKIPS=rowskips,
        )
    except (CommunicationError, LogonError, ABAPApplicationError, ABAPRuntimeError) as e:
        log.error("RFC_READ_TABLE error on %s: %s", table, e)
        raise

    cols = [f["FIELDNAME"] for f in res.get("FIELDS", [])]
    out: List[Dict[str, str]] = []
    for row in res.get("DATA", []):
        wa = row.get("WA", "")
        parts = wa.split(delimiter)
        if len(parts) < len(cols):
            parts += [""] * (len(cols) - len(parts))
        out.append({c: p.strip() for c, p in zip(cols, parts)})
    return out

