from setup import *
db = Oracle(database='WFOCUSP')

#################################### FLAGS ####################################

@dataclasses.dataclass
class Flags(MyBaseClass):
    root_path: str = "/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources/flags"
    def __post_init__(self):
        super().__post_init__()
        self.path = dict()
        for nm in ['raw','sheet','parq','csv']:
            self.path[nm] = self.root_path / nm

    def raw_to_parq(self, overwrite=False):
        for src in self.path['raw'].iterdir():
            src.replace(src.with_name(src.name.lower().replace(' ','_').replace('-','_')))
        k = 0
        for src in sorted(self.path['raw'].glob('*flags*.xlsx'), reverse=True):
            book = pd.ExcelFile(src, engine='openpyxl')
            try:
                dt = pd.to_datetime(src.stem[:10].replace('_','-'))
                sheets = {sheet:sheet for sheet in book.sheet_names if sheet.isnumeric() and int(sheet) % 100 in [1,6,8]}
            except:
                dt = pd.to_datetime(src.stem[-6:])
                sheets = {src.stem[:6]: book.sheet_names[0]}

            date = str(dt.date())
            for term_code, sheet in sheets.items():
                dst = self.path['sheet'] / f'{term_code}/flg_{term_code}_{date}.parq'
                if overwrite:
                    delete(dst)
                if not dst.is_file():
                    k = 0
                    mkdir(dst.parent)
                    df = book.parse(sheet)
                    df.insert(0, 'current_date', date)
                    write(dst, df)
                    delete(self.path['parq'] / f'flg_{term_code}.parq')
            k += 1
            if k > 9:
                break

    def combine(self, overwrite=False):
        col_drop = {
            '',
            'pidm_key',
            'street1_line1',
            'street1_line2',
            'unnamed:_0',
            'nvl((selectcasewhenp1.rcrapp1_',
            'nvl((selectcasewhenp1.rcrapp1_fed_coll_choice_1=:"sys_b_054"then:"sys_b_055"whenp1.rcrapp1_fed_coll_choice_2=:"sys_b_056"then:"s',
            'decisiondate',
        }

        col_repl = {
            'admt_code_desc': 'admt_desc',
            'admt_code_descr.':'admt_desc',
            'apdc':'apdc_code',
            'ap_decision':'apdc_desc',
            'app_decision':'apdc_desc',
            'decision_date':'apdc_date',
            'apdc_decision_date1': 'apdc_date',
            'apst':'apst_code',
            'app_status':'apst_desc',
            'ap_status_description': 'apst_desc',
            'campus name':'camp_desc',
            'campus_name':'camp_desc',
            'campus_code':'camp_code',
            'fafsa':'fafsa_app',
            'high_school':'hs_name',
            'sbgi_desc_high_school':'hs_name',
            'major_code':'majr_code',
            'major':'majr_desc',
            'majr_desc1':'majr_desc',
            'moa_hs':'mou_hs',
            'oren_sess':'orien_sess',
            'selected_for_verif':'selected_for_ver',
            'student_type':'styp_desc',
            'styp':'styp_code',
            'term_code_key':'term_code',
            'verif_complete':'ver_complete',
            'app_waiver_code':'waiver_code',
            'waiver_descriton':'waiver_desc',
            'tsi_hold_exists':'tsi_holds_exist',
            'zip1':'zip',
            'um_meningitis_hold_exists':'bacmen_hold_exists',
            'housing_hold_exists':'housing_holds',
        }

        for p in sorted(self.path['sheet'].iterdir(), reverse=True):
            term_code = p.stem
            dst = self.path['parq'] / f'flg_{term_code}.parq'
            csv = self.path['csv' ] / f'flg_{term_code}.csv'
            if overwrite:
                delete(dst)
            if not dst.is_file():
                print(dst)
                L = []
                for src in sorted(p.iterdir()):
                    df = read(src)
                    try:
                        col_repl['campus'] = 'camp_desc' if 'campus_code' in df else 'camp_code'
                    except:
                        df.disp(5)
                        assert 1==2
                    L.append(df.dropna(axis=1, how='all').drop(columns=col_drop, errors='ignore').rename(columns=col_repl, errors='ignore'))
                df = pd.concat(L)
                write(dst, pd.concat(L).prep())
                # delete(csv)
            # if not csv.is_file():
            #     print(csv)
            #     write(csv, read(dst), index=False)

    def completeness(self):
        L = []
        for parq in sorted(self.path['parq'].iterdir(), reverse=True):
            df = read(parq)
            L.append((100 * df.count() / df.shape[0]).round().rename(parq.stem[-6:]).to_frame().T.prep())
        return pd.concat(L)

    def run(self, overwrite=False):
        self.raw_to_parq(overwrite)
        self.combine(overwrite)
        # return self.completeness()

# #################################### Common ####################################
        
# @dataclasses.dataclass
# class Common(MyBaseClass):
#     root_path: str = "/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources/data"
#     def __post_init__(self):
#         super().__post_init__()
#         self.get('common')
    
#     def get_common(self):
#         qry = f"""
# select
#     A.stvterm_code as term_code,
#     replace(A.stvterm_desc, ' ', '') as term_desc,
#     A.stvterm_start_date as start_date,
#     A.stvterm_end_date as end_date,
#     A.stvterm_fa_proc_yr as fa_proc_yr,
#     A.stvterm_housing_start_date as housing_start_date,
#     A.stvterm_housing_end_date as housing_end_date,
#     B.sobptrm_census_date as census_date
# from stvterm A, sobptrm B
# where A.stvterm_code = B.sobptrm_term_code and B.sobptrm_ptrm_code='1'"""
#         self.trm = db.execute(qry)
#         self.dst = read(self.root_path / 'dst.parquet')
#         assert self.dst is not None, f"Can't find distances file - you probably have the wrong path. If you absolutely must recreate it, the original code is below. But it has not been tested since originally written in December 2023. It will almost surely have bugs that will require signficant effort to correct. You should exhaust every option to find the existing distance file before trying to run it."

        # import zipcodes, openrouteservice
        # client = openrouteservice.Client(key=os.environ.get('OPENROUTESERVICE_API_KEY1'))
        # def get_distances(Z, eps=0):
        #     theta = np.random.rand(len(Z))
        #     Z = Z + eps * np.array([np.sin(theta), np.cos(theta)]).T
        #     L = []
        #     dk = 1000 // len(dst)
        #     k = 0
        #     while k < len(Z):
        #         print(f'getting distances {rjust(k,5)} / {len(Z)} = {rjust(round(k/len(Z)*100),3)}%')
        #         src = Z.iloc[k:k+dk]
        #         X = pd.concat([dst,src]).values.tolist()
        #         res = client.distance_matrix(X, units="mi", destinations=list(range(0,len(dst))), sources=list(range(len(dst),len(X))))
        #         L.append(pd.DataFrame(res['durations'], index=src.index, columns=camp.keys()))
        #         k += dk
        #     return pd.concat(L)

        # Z = [[z['zip_code'],z['state'],z['long'],z['lat']] for z in zipcodes.list_all() if z['state'] not in ['PR','AS','MH','PW','MP','FM','GU','VI','AA','HI','AK','AP','AE']]
        # Z = pd.DataFrame(Z, columns=['zip','state','lon','lat']).prep().query('lat>20').set_index(['zip','state']).sort_index()
        # camp = {'s':76402, 'm':76036, 'w':76708, 'r':77807, 'l':76065}
        # dst = Z.loc[camp.values()]
        # try:
        #     df = read(fn)
        # except:
        #     df = get_distances(Z)
        # for k in range(20):
        #     mask = df.isnull().any(axis=1)
        #     if mask.sum() == 0:
        #         break
        #     df = df.combine_first(get_distances(Z.loc[mask], 0.02*k))
        # return write(fn, df.assign(o = 0).melt(ignore_index=False, var_name='camp_code', value_name='distance').prep())