
SAP_DEFAULT_SYSTEM = "K11"

SNC_PARTNER = " "

MSHOST_P11 = "rffsp11s.sap.roto-frank.com"
MSSERV_P11 = "sapmsP11"
LOGON_GROUP = "ROTO_FRANK"
CLIENT = "151"
LANG = "PL"

MSHOST_K11 = "rffsk11s.sap.roto-frank.com"
MSSERV_K11 = "sapmsK11"

SAP_SYSTEMS = {
    "K11": {
        "description": "System testowy K11 - logowanie hasłem",
        "connection": {
            "mshost": MSHOST_K11,
            "msserv": "3602",
            "sysid":  "K11",
            "group":  LOGON_GROUP,
            "client": CLIENT,
            "lang":   LANG,
            "snc_mode": "1",
            "snc_qop":  "8",
            "snc_partnername": SNC_PARTNER,
        },
    },
    "P11_SSO": {
        "description": "System produkcyjny P11 - logowanie SSO/SNC",
        "connection": {
            "mshost": MSHOST_P11,
            "msserv": MSSERV_P11,
            "sysid":  "P11",
            "group":  LOGON_GROUP,
            "client": CLIENT,
            "lang":   LANG,
            "snc_mode": "1",
            "snc_qop":  "8",
            "snc_partnername": SNC_PARTNER,
        },
    },
}
