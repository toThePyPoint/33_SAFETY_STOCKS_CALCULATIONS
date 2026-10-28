#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Dict, Any

from pyrfc import (
    Connection,
    CommunicationError,
    LogonError,
    ABAPRuntimeError,
    ABAPApplicationError,
)

from sap_config import SAP_SYSTEMS, SAP_DEFAULT_SYSTEM
from log_utils import setup_logger

log = setup_logger("SAP_CONN", "sap_conn.log")


def build_sap_params(system: Optional[str] = None) -> Dict[str, Any]:
    if not SAP_SYSTEMS:
        raise RuntimeError("SAP_SYSTEMS w sap_config.py jest puste!")

    if system is None:
        system = SAP_DEFAULT_SYSTEM

    if system not in SAP_SYSTEMS:
        raise RuntimeError(f"Nieznany system SAP: {system!r}")

    entry = SAP_SYSTEMS[system]
    conn_cfg = entry.get("connection")
    if not isinstance(conn_cfg, dict):
        raise RuntimeError(f"Brak sekcji 'connection' dla systemu {system!r}")

    params: Dict[str, Any] = dict(conn_cfg)
    params.setdefault("lang", "PL")
    
    params.setdefault("timeout", '300') 

    safe_params = {
        k: ("***" if k.lower() in ("passwd", "password") else v)
        for k, v in params.items()
    }
    log.info("Buduję parametry dla systemu %s: %s", system, safe_params)

    return params


@contextmanager
def get_conn(system: Optional[str] = None) -> Connection:
    params = build_sap_params(system)
    conn: Optional[Connection] = None
    system_name = system or SAP_DEFAULT_SYSTEM

    try:
        conn = Connection(**params)

        if 'passwd' in params:
            params['passwd'] = "CLEARED"
        del params

        log.info("Połączenie z SAP nawiązane (system=%s).", system_name)
        yield conn

    except (CommunicationError, LogonError, ABAPApplicationError, ABAPRuntimeError) as e:
        log.error("Błąd połączenia z SAP (system=%s): %s", system_name, e)
        raise

    finally:
        if conn is not None:
            try:
                if getattr(conn, "alive", False):
                    conn.close()
                    log.info("Połączenie SAP zamknięte (system=%s).", system_name)
            except Exception as e:
                log.warning("Błąd przy zamykaniu połączenia SAP: %s", e)
