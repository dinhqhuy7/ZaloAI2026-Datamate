"""NGANH for sinh.py = the 27 industries of nhanld (nganh_nhanld.py) + 15 new industries (nganh_moi.py)."""
from nganh_moi import NGANH_MOI
from nganh_nhanld import NGANH as _NGANH_NHANLD

NGANH = {**_NGANH_NHANLD, **NGANH_MOI}
