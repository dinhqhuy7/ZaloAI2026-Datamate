"""Create the shop description file (industry, shop type, ledger style, all features) for the 6 parts of Data_Zalo/Gen_1311. A rerun gives the same output."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nganh import NGANH          # noqa: E402  (the industry store already exists: used for the ledger-style and rule-change parts)

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_DO = {'kinh_mat', 'gao', 'son', 'mo_hinh', 'hai_san_kho', 'len_soi'}
CU = sorted(set(NGANH) - GEN_DO)


def ghi(ten, ds):
    json.dump(ds, open(os.path.join(HERE, ten + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(ten, len(ds))


LAY4 = ['phang', 'csv_tach', 'excel_thang', 'sqlite', 'file_thang', 'xlsx_nhieu_file']      # ledger styles that have product rows

# 1. new industries: 10 industries, 5 industries of 3 shops + 5 industries of 2 shops; ledgers with product rows; questions concentrated on quantity, unit, conversion, product name
R = random.Random('1311-1')
TANG1 = {'qty_item': 3, 'units': 3, 'top_item': 2, 'rev_item': 2, 'top_rev_group': 2, 'group': 1.5, 'hike': 1.5}
moi = ['trung_gia_cam', 'gas_bep', 'nuoc_da_binh', 'gach_op_lat', 'dau_nhot', 'vang_bac', 'do_nhua', 'nhom_kinh', 'than_cui', 'hoa_pham']
ds = []
for i, n in enumerate(moi):
    ds += [{'nganh': n, 'kieu': '', 'layout': R.choice(LAY4), 'tang_ho': TANG1} for _ in range(3 if i < 5 else 2)]
ghi('nganh_moi', ds)

# 2. five new ledger styles x 3 shops; industry ledgers: vehicle repair, phones, mother and baby (loyalty points)
R = random.Random('1311-2')
LAY = ['sqlite_pos', 'tron_nguon', 'so_chuyen_nganh', 'ghi_tay_so_phu', 'mot_sheet_khong_ma']
cn = [('sua_xe_may', 'sua_xe'), ('dien_thoai_may', 'dien_thoai'), ('me_be', 'diem')]
nn = R.sample([k for k in CU if k != 'me_be'], 12)
ds = []
for i in range(15):
    lay = LAY[i % 5]
    if lay == 'so_chuyen_nganh':
        n, k = cn[i // 5]
        ds.append({'nganh': n, 'kieu': k, 'layout': lay, 'giau': True})
    else:
        ds.append({'nganh': nn.pop(), 'kieu': '', 'layout': lay, 'giau': lay in ('sqlite_pos', 'tron_nguon') or R.random() < 0.5})
ghi('kieu_so_moi', ds)

# 3. industry-specific rules (§6); the ledger must have product rows (drug batches, labour charge, IMEI live on the rows); questions concentrated on the families under the specific rules
R = random.Random('1311-3')
TANG3 = {'vet': 7, 'coc': 7, 'debt': 2, 'sua_xe': 6, 'dien_thoai': 6, 'diem': 7}
ds = [{'nganh': 'nha_thuoc', 'kieu': 'rx_lo', 'layout': R.choice(LAY4), 'giau': True, 'tang_ho': TANG3} for _ in range(3)]
ds += [{'nganh': 'tiem_may', 'kieu': 'coc', 'layout': R.choice(LAY4), 'giau': False, 'tang_ho': TANG3},
       {'nganh': 'banh_kem', 'kieu': 'coc', 'layout': '', 'giau': False, 'tang_ho': TANG3}]
ds += [{'nganh': 'sua_xe_may', 'kieu': 'sua_xe', 'layout': R.choice(LAY4), 'giau': True, 'tang_ho': TANG3} for _ in range(2)]
ds += [{'nganh': 'dien_thoai_may', 'kieu': 'dien_thoai', 'layout': R.choice(LAY4), 'giau': True, 'tang_ho': TANG3},
       {'nganh': 'me_be', 'kieu': 'diem', 'layout': R.choice(LAY4), 'giau': True, 'tang_ho': TANG3}]
ds += [{'nganh': 'tram_sac', 'kieu': 'sac'} for _ in range(2)]
ds += [{'nganh': n, 'kieu': 'thue'} for n in ('thue_do_cuoi', 'thue_xe_may', 'thue_cam_trai')]
ghi('luat_nganh', ds)

# 4. rule changes E / F / G: 5 shops per variant, all features (on-credit, delivery), ledgers with product rows
R = random.Random('1311-4')
for L in 'EFG':
    ds = [{'nganh': n, 'kieu': '', 'layout': R.choice(LAY4), 'giau': True} for n in R.sample(CU, 5)]
    ghi('doi_luat_' + L, ds)
