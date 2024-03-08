from setup import *
import traceback
db = Oracle('WFOCUSP')
root_path = pathlib.Path("/home/scook/institutional_data_analytics/admitted_matriculation_projection/LiveAMP")

def dt(date, format='sql'):
    try:
        date = pd.to_datetime(date).date()
    except:
        return f"trunc({date})"
    format = format.lower()
    if format == 'sql':
        d = date.strftime('%d-%b-%y')
        return f"trunc(to_date('{d}'))"
    elif format == 'iso':
        return date.isoformat()
    else:
        return date.strftime(format)

def get_term(term_code, attr='desc'):
    if attr in ['desc','start_date','end_date','fa_proc_yr','housing_start_date','housing_end_date']:
        qry = f"select stvterm_{attr} from stvterm where stvterm_code = {term_code}"
    elif attr in ['census_date']:
        qry = f"select sobptrm_{attr} from sobptrm where sobptrm_term_code = {term_code} and sobptrm_ptrm_code = '1'"
    else:
        raise Exception(f'unknown term attr: {attr}')
    x = db.execute(qry).squeeze()
    return x.replace(' ','') if isinstance(x, str) else x

def get_desc(nm, alias=None):
    tbl = 'stv'+nm if alias is None else 'stv'+alias
    return [f"A.{nm}_code", f"(select B.{tbl}_desc from {tbl} B where B.{tbl}_code = A.{nm}_code) as {nm}_desc"]

#################################### FLAGS ####################################

@dataclasses.dataclass
class FLAGS(MyBaseClass):
    def __post_init__(self):
        self.path = {'root': root_path / 'flags'}
        self.path['raw'] = self.path['root'] / 'raw'
        self.path['sheet'] = self.path['root'] / 'sheet'
        self.path['parq'] = self.path['root'] / 'parq'
        self.path['csv']  = self.path['root'] / 'csv'

    def raw_to_parq(self, overwrite=False):
        for src in self.path['raw'].iterdir():
            src.replace(src.with_stem(src.stem.lower().replace(' ','_').replace('-','_')))
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
                    print(dst)
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
                    col_repl['campus'] = 'camp_desc' if 'campus_code' in df else 'camp_code'
                    L.append(df.dropna(axis=1, how='all').drop(columns=col_drop, errors='ignore').rename(columns=col_repl, errors='ignore'))
                write(dst, pd.concat(L))
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

#################################### TERM ####################################

@dataclasses.dataclass
class TERM(MyBaseClass):
    term_code: int
    cycle_day: int = 0
    overwrite: typing.Dict = None
    show: typing.Dict = None

    def __post_init__(self):
        # D = {'adm':False, 'reg':False, 'flg':False, 'raw':False}
        # for x in ['overwrite','show']:
        #     self[x] = D.copy() if self[x] is None else D.copy() | self[x]
        self.year = self.term_code // 100
        self.term = self.term_code % 100
        self.appl_term_code = [self.term_code-2, self.term_code] if self.term == 8 else [self.term_code]
        self.appl_term_desc = [get_term(t,'desc') for t in self.appl_term_code]
        self.term_desc = get_term(self.term_code, 'desc')
        self.end_date = get_term(self.term_code, 'census_date') + pd.Timedelta(days=7)
        self.cycle_date = self.end_date - pd.Timedelta(days=self.cycle_day)
        self.path = {'root': root_path}
        self.path['data'] = self.path['root'] / f"data"
        self.path['adm']  = self.path['data'] / f"adm/{self.term_code}"
        self.path['reg']  = self.path['data'] / f"reg/{self.term_code}"
        self.path['flg']  = self.path['data'] / f"flg/{self.term_code}"
        self.path['raw']  = self.path['data'] / f"raw/{self.term_code}"
        self.flg_col = {
            'perm': [
                'id',
                'styp_code',
                'current_date',
                # 'transfer_hours',
                # 'inst_hours',
                # 'overall_hours',
                # 'inst_gpa',
                # 'overall_gpa',
                'fafsa_app',
                # 'finaid_offered',
                'finaid_accepted',
                # 'finaid_declined',
                'disb_req_complete',
                'schlship_app',
                # 'declined_schlrship',
                'math',
                'reading',
                'writing',
                'gap_score',
                ],
            'temp': [
                'app_date',
                'term_code',
                'ssb_last_accessed',
                'waiver_code',
                'ftic_gap_score',
                't_gap_score',
                'orien_sess',
                'orientation_hold_exists',
                'registered',
                'ver_complete',
                'selected_for_ver',
                'act_new_comp_score',
                'sat10_total_score',
                ],
        }

    def run(self, qry, fn=None, show=False, binarize=False):
        df = db.execute(qry, show=show)
        if binarize:
            df = df.binarize()
        if fn is not None:
            write(fn, df, overwrite=True)
        return df

    def get_dst(self, cycle_day=None):
        fn = self.path['root'] / f"distances.parq"
        df = read(fn)
        if df is not None:
            return df
        print(f'creating {fn.name}')
        import zipcodes, openrouteservice
        client = openrouteservice.Client(key=os.environ.get('OPENROUTESERVICE_API_KEY1'))
        def get_distances(Z, eps=0):
            theta = np.random.rand(len(Z))
            Z = Z + eps * np.array([np.sin(theta), np.cos(theta)]).T
            L = []
            dk = 1000 // len(dst)
            k = 0
            while k < len(Z):
                print(f'getting distances {rjust(k,5)} / {len(Z)} = {rjust(round(k/len(Z)*100),3)}%')
                src = Z.iloc[k:k+dk]
                X = pd.concat([dst,src]).values.tolist()
                res = client.distance_matrix(X, units="mi", destinations=list(range(0,len(dst))), sources=list(range(len(dst),len(X))))
                L.append(pd.DataFrame(res['durations'], index=src.index, columns=camp.keys()))
                k += dk
            return pd.concat(L)

        Z = [[z['zip_code'],z['state'],z['long'],z['lat']] for z in zipcodes.list_all() if z['state'] not in ['PR','AS','MH','PW','MP','FM','GU','VI','AA','HI','AK','AP','AE']]
        Z = pd.DataFrame(Z, columns=['zip','state','lon','lat']).prep().query('lat>20').set_index(['zip','state']).sort_index()
        camp = {'s':76402, 'm':76036, 'w':76708, 'r':77807, 'l':76065}
        dst = Z.loc[camp.values()]
        try:
            df = read(fn)
        except:
            df = get_distances(Z)
        for k in range(20):
            mask = df.isnull().any(axis=1)
            if mask.sum() == 0:
                break
            df = df.combine_first(get_distances(Z.loc[mask], 0.02*k))
        return write(fn, df.assign(o = 0).melt(ignore_index=False, var_name='camp_code', value_name='distance').prep())
    
    def flg_fill(self):
        F = dict()
        for fn in sorted(self.path['flg'].iterdir()):
            term_code = int(fn.stem[-6:])
            if term_code >= 202006:
                F[term_code] = read(fn).notnull().mean()*100
        return pd.DataFrame(F).round().prep()

    def get_cycle_day(self, col='A.current_date'):
        return f"{dt(self.end_date)} - trunc({col})"
    
    def cutoff(self, col='A.current_date', criteria="= 0"):
        return f'{self.get_cycle_day(col)} {criteria}'
    
    def prep(self, nm, cycle_day=None):
        cycle_day = self.cycle_day if cycle_day is None else cycle_day
        fn = self.path[nm] / f"{nm}_{self.term_code}_{rjust(cycle_day,3,0)}.parq"
        df = read(fn, self.overwrite[nm])
        # self.overwrite[nm] = False
        if df is None:
            print(f'{fn.name} not found - creating')
        return cycle_day, fn, df

#     def get_reg(self, cycle_day=None):
#         nm = 'reg'
#         cycle_day, fn, df = self.prep(nm, cycle_day)
#         if df is not None:
#             return df
#         try:
#             db.head(f'opeir.registration_{self.term_desc}')
#         except:
#             return pd.DataFrame(columns=['current_date','cycle_day','term_code','pidm','id','levl_code','styp_code','crn','crse','credit_hr'])

#         sel = join([
#             # f"{self.get_cycle_day()} as cycle_day",
#             # f"trunc(A.current_date) as cycle_date",
#             f"{self.term_code} as term_code",
#             # f"min(case when {self.get_cycle_day()} >= {cycle_day} then {self.get_cycle_day()} end) over (partition by A.pidm, A.crn) as before",
#             # f"min({self.get_cycle_day()}) over (partition by A.pidm, A.crn) as after",
#             f"A.pidm",
#             # f"A.id",
#             f"A.levl_code",
#             f"A.styp_code",
#             # f"A.crn",
#             f"lower(A.subj_code) || A.crse_numb as crse",
#             f"A.credit_hr",
#         ], C+N)
#         qry = f"""
# select {indent(sel)}
# from opeir.registration_{self.term_desc} A
# where
#     {self.get_cycle_day()} - {cycle_day} between 0 and 14
#     and A.credit_hr > 0
#     and A.subj_code <> 'INST'"""

#         sel = join([
#             f"A.cycle_day",
#             # f"A.cycle_date",
#             f"term_code",
#             f"A.pidm",
#             f"A.id",
#             f"levl_code",
#             f"styp_code",
#             f"A.crn",
#             f"A.crse",
#             f"A.credit_hr",
#         ], C+N)

#         qry = f"""
# select {indent(sel)}
# from {subqry(qry)} A
# where
#     A.cycle_day = A.before
#     and (A.after <= {cycle_day} or sysdate - {dt(self.cycle_date)} < 5)
#     --and A.levl_code = 'UG'
#     --and A.styp_code in ('N','R','T')"""

        # return self.run(qry, fn, self.show[nm])


#     def get_reg(self, cycle_day=None):
#         nm = 'reg'
#         cycle_day, fn, df = self.prep(nm, cycle_day)
#         if df is not None:
#             return df
#         try:
#             db.head(f'opeir.registration_{self.term_desc}')
#         except:
#             return pd.DataFrame(columns=['current_date','cycle_day','term_code','pidm','id','levl_code','styp_code','crn','crse','credit_hr'])

#         sel = join([
#             f"{cycle_day} as cycle_day",
#             # f"trunc(A.current_date) as cycle_date",
#             f"{self.term_code} as term_code",
#             f"A.pidm",
#             f"max(A.levl_code) as levl_code",
#             f"max(A.styp_code) as styp_code",
#             f"lower(A.subj_code) || A.crse_numb as crse",
#             f"max(A.credit_hr) as credit_hr",
#         ], C+N)
#         qry = f"""
# select {indent(sel)}
# from opeir.registration_{self.term_desc} A
# where
#     {self.get_cycle_day()} - {cycle_day} between 0 and 14
#     and A.credit_hr > 0
#     and A.subj_code <> 'INST'
# group by A.pidm, A.subj_code, A.crse_numb"""
        
#         qry = f"""
# with A as {subqry(qry)}
# select A.* from A
# union all
# select
#     A.cycle_day, A.term_code, A.pidm, A.levl_code, A.styp_code,
#     '_total' as crse,
#     sum(A.credit_hr) as credit_hr
# from A
# group by A.cycle_day, A.term_code, A.pidm, A.levl_code, A.styp_code"""

#         return self.run(qry, fn, self.show[nm])



    def get_reg(self, cycle_day=None):
        nm = 'reg'
        cycle_day, fn, df = self.prep(nm, cycle_day)
        if df is not None:
            return df
        try:
            db.head(f'opeir.registration_{self.term_desc}')
        except:
            return pd.DataFrame(columns=['current_date','cycle_day','term_code','pidm','id','levl_code','styp_code','crn','crse','credit_hr'])

        qry = f"""
select
    {cycle_day} as cycle_day,
    A.sfrstcr_term_code as term_code,
    A.sfrstcr_pidm as pidm,
    (select C.sgbstdn_levl_code from sgbstdn C where C.sgbstdn_pidm = A.sfrstcr_pidm and C.sgbstdn_term_code_eff <= A.sfrstcr_term_code order by C.sgbstdn_term_code_eff desc fetch first 1 rows only) as levl_code,
    (select C.sgbstdn_styp_code from sgbstdn C where C.sgbstdn_pidm = A.sfrstcr_pidm and C.sgbstdn_term_code_eff <= A.sfrstcr_term_code order by C.sgbstdn_term_code_eff desc fetch first 1 rows only) as styp_code,
    lower(B.ssbsect_subj_code) || B.ssbsect_crse_numb as crse,
    sum(B.ssbsect_credit_hrs) as credit_hr
from sfrstcr A, ssbsect B
where
    A.sfrstcr_term_code = B.ssbsect_term_code
    and A.sfrstcr_crn = B.ssbsect_crn
    and A.sfrstcr_term_code = {self.term_code}
    and A.sfrstcr_ptrm_code not in ('28','R3')
    and  {self.get_cycle_day('A.sfrstcr_add_date')} >= {cycle_day}  -- added before cycle_day
    and ({self.get_cycle_day('A.sfrstcr_rsts_date')} < {cycle_day} or A.sfrstcr_rsts_code in ('DC','DL','RD','RE','RW','WD','WF')) -- dropped after cycle_day or still enrolled
    and B.ssbsect_subj_code <> 'INST'
group by A.sfrstcr_term_code, A.sfrstcr_pidm, B.ssbsect_subj_code, B.ssbsect_crse_numb
"""

        qry = f"""
with A as {subqry(qry)}
select A.* from A
union all
select
    A.cycle_day, A.term_code, A.pidm, A.levl_code, A.styp_code,
    '_total' as crse,
    sum(A.credit_hr) as credit_hr
from A
group by A.cycle_day, A.term_code, A.pidm, A.levl_code, A.styp_code
"""




#         qry = f"""
# select
#     sfrstcr_term_code as term_code,
#     sfrstcr_pidm as pidm,
#     (select B.sgbstdn_levl_code from sgbstdn B where B.sgbstdn_pidm = A.sfrstcr_pidm and B.sgbstdn_term_code_eff <= A.sfrstcr_term_code order by B.sgbstdn_term_code_eff desc fetch first 1 rows only) as styp_code,
#     (select B.sgbstdn_styp_code from sgbstdn B where B.sgbstdn_pidm = A.sfrstcr_pidm and B.sgbstdn_term_code_eff <= A.sfrstcr_term_code order by B.sgbstdn_term_code_eff desc fetch first 1 rows only) as levl_code,
#     --lower(B.ssbsect_subj_code) as subj_code,
#     --B.ssbsect_crse_numb as crse_numb,
#     lower(B.ssbsect_subj_code) || B.ssbsect_crse_numb as crse,
#     {self.get_cycle_day('A.sfrstcr_add_date')} as add_day,
#     {self.get_cycle_day('A.sfrstcr_rsts_date')} as rsts_day,
#     A.sfrstcr_rsts_code as rsts_code,
#     B.ssbsect_credit_hrs as credit_hr
# from sfrstcr A, ssbsect B
# where
#     A.sfrstcr_term_code = B.ssbsect_term_code
#     and A.sfrstcr_crn = B.ssbsect_crn
#     and A.sfrstcr_term_code = {self.term_code}
#     and A.sfrstcr_ptrm_code not in ('28','R3')
#     --and A.sfrstcr_credit_hr > 0
#     and {self.get_cycle_day('A.sfrstcr_add_date')} >= {cycle_day}  -- added before cycle_day
#     and (A.sfrstcr_rsts_code in ('DC','DL','RD','RE','RW','WD','WF') or {self.get_cycle_day('A.sfrstcr_rsts_date')} < {cycle_day})  -- still enrolled or dropped after cycle_day
#     and B.ssbsect_subj_code <> 'INST'
# """


        return self.run(qry, fn, self.show[nm])


    def get_adm(self, cycle_day=None):
        nm = 'adm'
        cycle_day, fn, df = self.prep(nm, cycle_day)
        if df is not None:
            return df

        def f(term_desc):
            accept = "A.apst_code = 'D' and A.apdc_code in (select stvapdc_code from stvapdc where stvapdc_inst_acc_ind is not null)"
            reject = "(A.apst_code in ('X', 'W')) or (A.apst_code = 'D' and (substr(A.apdc_code,1,1) in ('D','W') or A.apdc_code = 'RJ'))"
            sel = join([
                f"{self.get_cycle_day()} as cycle_day",
                f"trunc(A.current_date) as cycle_date",
                f"min(trunc(A.current_date)) over (partition by A.pidm, A.appl_no) as appl_date",  # first date on snapshot table (saradap_appl_date has too many consistencies so this replaces it)
                f"min(case when {accept} then trunc(A.current_date) end) over (partition by A.pidm, A.appl_no) as apdc_date",  # first date accepted
                f"A.pidm",
                f"A.id",
                f"A.term_code as term_code_entry",
                f"A.levl_code",
                f"A.styp_code",
                f"A.admt_code",
                f"A.appl_no",
                f"A.apst_code",
                f"A.apdc_code",
                f"A.camp_code",
                f"A.saradap_resd_code as resd_code",
                # f"A.gender",
                # f"A.birth_date",
                f"A.coll_code_1 as coll_code",
                f"A.majr_code_1 as majr_code",
                f"A.dept_code",
                f"A.hs_percentile as hs_pctl",
                # f"case{N+T}when {reject} then -1{N+T}when {accept} then 1{N+T}else 0 end as status",
                # f"case{N+T}when max(case when {self.get_cycle_day()} >= {cycle_day} then A.current_date end) over (partition by A.pidm, A.appl_no) = A.current_date then 1{N+T}end as r1",  # finds most recent daily snapshot BEFORE cycle_day
                # f"case{N+T}when sum(case when {self.get_cycle_day()} between 0 and {cycle_day}-1 then 1 end) over (partition by A.pidm, A.appl_no) >= {cycle_day}/2 then 1{N+T}when sysdate - {dt(self.cycle_date)} < 5 then 1{N+T}end as r2",  # check if appears in >= 50% of daily snapshots AFTER cycle_day
            ], C+N)

            qry = f"select {indent(sel)}{N}from opeir.admissions_{term_desc} A"
            
            sel = join([
                f"A.*",
                f"case{N+T}when max(case when A.cycle_day >= {cycle_day} then A.cycle_date end) over (partition by A.pidm, A.appl_no) = A.cycle_date then 1{N+T}end as r1",  # finds most recent daily snapshot BEFORE cycle_day
                f"case{N+T}when sum(case when A.cycle_day <  {cycle_day} then 1 end) over (partition by A.pidm, A.appl_no) >= {cycle_day}/2 then 1{N+T}when sysdate - {dt(self.cycle_date)} < 5 then 1{N+T}end as r2",  # check if appears on >= 50% of daily snapshots AFTER cycle_day
            ], C+N)

            qry = f"select {indent(sel)}{N}from {subqry(qry)} A where cycle_day between 0 and {cycle_day} + 14 and {accept}"
            qry = f"select A.* from {subqry(qry)} A where A.r1 = 1 and A.r2 = 1"
            return qry
        qry = join([f(term_desc).strip() for term_desc in self.appl_term_desc], "\n\nunion all\n\n")
        # qry = f"select A.*, row_number() over (partition by A.pidm order by A.appl_no desc) as r from {subqry(qry)} A"

#  where A.status = 1 and A.r1 = 1 and A.r2 = 1"
#     *
# from 
           

#             f"case{N+T}when max(case when {self.get_cycle_day()} >= {cycle_day} then A.current_date end) over (partition by A.pidm, A.appl_no) = A.current_date then 1{N+T}end as r1",  # finds most recent daily snapshot BEFORE cycle_day
#             f"case{N+T}when sum(case when {self.get_cycle_day()} between 0 and {cycle_day}-1 then 1 end) over (partition by A.pidm, A.appl_no) >= {cycle_day}/2 then 1{N+T}when sysdate - {dt(self.cycle_date)} < 5 then 1{N+T}end as r2",  # check if appears in >= 50% of daily snapshots AFTER cycle_day




# where {self.get_cycle_day()} between 0 and {cycle_day}+14"""
#             return f"select A.* from {subqry(qry)} A where A.r1 = 1 and A.r2 = 1 and {accept}"
#             # return f"select A.* from {subqry(qry)} A where A.status = 1 and A.r1 = 1 and A.r2 = 1"

        # qry = f"""select A.*, row_number() over (partition by A.pidm order by A.status desc, A.appl_no desc) as r from {subqry(frm)} A"""

#         qry = f"""
# select
#     A.*,
#     min(A.current_date) over (partition by A.pidm, A.appl_no) as first_date,
#     min(case when {accept} then A.current_date end) over (partition by A.pidm, A.appl_no) as accept_date,
#     case when max(case when cycle_day >= {cycle_day} then A.current_date end) over (partition by A.pidm, A.appl_no) = A.current_date then 1 end as q1,
#     case when sum(case when cycle_day between 0 and {cycle_day}-1 then 1 end) over (partition by A.pidm, A.appl_no) >= {cycle_day}/2 then 1
#          when sysdate - {dt(self.cycle_date)} < 5 then 1 end as q2
# from {subqry(qry)} A"""

        # qry = f"""select A.*, row_number() over (partition by A.pidm order by A.status desc, A.appl_no) as r from {subqry(qry)} A"""

# A.before = A.cycle_rel and (A.after < 0 or sysdate - {dt(self.cycle_date)} < 5)"""



#         f = lambda term_desc: f"""
# select {indent(sel)}
# from opeir.admissions_{term_desc} A
# where {self.get_cycle_day()} - {cycle_day} between -14 and 14"""
        
#         qry = f"""
# select
#     A.*,
#     min(case when A.cycle_rel >= 0 then A.cycle_rel end) over (partition by A.pidm, A.appl_no) as before,
#     max(case when A.cycle_rel <  0 then A.cycle_rel end) over (partition by A.pidm, A.appl_no) as after
# from {subqry(qry)} A where A.cycle_rel between -14 and 14"""
        
#         qry = f"""
# select
#     A.*,
#     row_number() over (partition by A.pidm order by A.status desc, A.appl_no) as r
# from {subqry(qry)} A where A.before = A.cycle_rel and (A.after < 0 or sysdate - {dt(self.cycle_date)} < 5)"""
        

        def get_spraddr(nm):
            if nm == 'zip':
                sel = "to_number(substr(B.spraddr_zip, 0, 5) default null on conversion error)"
            else:
                sel = "B.spraddr_"+nm
            return f"(select {sel} from spraddr B where B.spraddr_pidm = A.pidm and B.spraddr_atyp_code in ('PA','MA','BU','BI') order by B.spraddr_atyp_code desc, B.spraddr_seqno desc fetch first 1 row only) as {nm}"

        sel = join([
            f"A.*",
            f"row_number() over (partition by A.pidm order by A.appl_no desc) as r",
            f"{self.term_code} as term_code",
            get_spraddr("cnty_code"),
            get_spraddr("stat_code"),
            get_spraddr("natn_code"),
            get_spraddr("zip"),
            f"(select B.spbpers_lgcy_code from spbpers B where B.spbpers_pidm = A.pidm) as lgcy_code",
            f"(select B.spbpers_birth_date from spbpers B where B.spbpers_pidm = A.pidm) as birth_date",
            # f"case when A.coll_code = 'AE' then 'AN' when A.coll_code = 'EH' then 'ED' when A.coll_code = 'HS' then 'HL' when A.coll_code = 'ST' then 'SM' else A.coll_code end as coll_code",
        ], C+N)
        qry = f"select {indent(sel)}{N}from {subqry(qry)} A"
        
        def coal(x):
            s = ' as '
            y, z = x.split(s)
            return f"coalesce({y}, 0){s}{z}"

        sel = N+T+join([
            f"A.cycle_day",
            f"{self.get_cycle_day('apdc_date')} as apdc_day",
            f"{self.get_cycle_day('appl_date')} as appl_day",
            f"{self.get_cycle_day('birth_date')} as birth_day",
            f"{dt(self.end_date)} as end_date",
            f"A.cycle_date",
            f"A.apdc_date",
            f"A.appl_date",
            f"A.birth_date",
            f"A.term_code_entry",
            *get_desc('term'),
            f"A.pidm",
            f"A.id",
            f"A.appl_no",
            *get_desc('levl'),
            *get_desc('styp'),
            *get_desc('admt'),
            # f"A.status",
            *get_desc('apst'),
            *get_desc('apdc'),
            *get_desc('camp'),
            f"case when A.camp_code = 'S' then 1 else 0 end as camp_main",
            *get_desc('cnty'),
            *get_desc('stat'),
            f"A.zip",
            f"A.natn_code",
            f"(select B.stvnatn_nation from stvnatn B where B.stvnatn_code = A.natn_code) as natn_desc",
            *get_desc('resd'),
            f"case when A.resd_code = 'R' then 1 else 0 end as resd",
            *get_desc('coll'),
            *get_desc('dept'),
            *get_desc('majr'),
            f"(select B.spbpers_sex from spbpers B where B.spbpers_pidm = A.pidm) as gender",
            *get_desc('lgcy'),
            f"case when A.lgcy_code is null or A.lgcy_code in ('N','O') then 0 else 1 end as legacy",
            coal(f"(select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='IN') as race_american_indian"),
            coal(f"(select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='AS') as race_asian"),
            coal(f"(select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='BL') as race_black"),
            coal(f"(select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='HA') as race_pacific"),
            coal(f"(select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='WH') as race_white"),
            coal(f"(select distinct 1 from spbpers B where B.spbpers_pidm = A.pidm and B.spbpers_ethn_cde=2   ) as race_hispanic"),
            # f"(select B.spbpers_ethn_cde-1 from spbpers B where B.spbpers_pidm = A.pidm) as race_hispanic",
            f"A.hs_pctl"
        ], C+N)
        qry = f"select {indent(sel)}\nfrom {subqry(qry)} A where A.r = 1 and A.levl_code = 'UG' and A.styp_code in ('N','R','T')"
        return self.run(qry, fn, self.show[nm], binarize=True)


    def get_flg(self, cycle_day=None):
        nm = 'flg'
        cycle_day, fn, df = self.prep(nm, cycle_day)
        if df is not None:
            return df
        F = []
        for term_code in self.appl_term_code:
            raw = FLAGS().path['parq'] / f"flg_{term_code}.parq"
            df = read(raw, columns=['current_date'])
            df['cycle_day'] = (self.end_date - df['current_date']).dt.days
            flg_day  = df.query(f'cycle_day>={cycle_day}')['cycle_day'].min()
            flg_date = df.query(f'cycle_day==@flg_day')['current_date'].min()
            filters = [('current_date','==',flg_date)]
            L = []
            missing = []
            if self.flg_col is None:
                df = read(raw, filters=filters)
            else:
                for c in sum(self.flg_col.values(),[]):
                    x = read(raw, filters=filters, columns=[c])
                    if x is None:
                        missing.append(c)
                        x = L[0][[]].assign(**{c:pd.NA})
                    L.append(x)
                df = pd.concat(L, axis=1)
            print(f'{term_code} flags cycle day {flg_day} >= {cycle_day} on {flg_date} missing columns: {missing}')
            F.append(df)
        with warnings.catch_warnings(action='ignore'):
            subset = ['id','term_code_entry','styp_code']
            df = (
                pd.concat(F, ignore_index=True)
                .rename(columns={'current_date':'flg_date', 'term_code':'term_code_entry'})
                .sort_values(by=[*subset,'app_date'])
                .drop_duplicates(subset=subset, keep='last')
                .copy()
                .prep()
            )
        df['gap_score'] = np.where(df['styp_code']=='n', df['ftic_gap_score'].combine_first(df['t_gap_score']).combine_first(df['gap_score']), df['t_gap_score'].combine_first(df['ftic_gap_score']).combine_first(df['gap_score']))
        df['ssb'] = df['ssb_last_accessed'].notnull()
        df['waiver'] = df['waiver_code'].notnull()
        df['oriented'] = np.where(df['orien_sess'].notnull() | df['registered'].notnull(), 'y', np.where(df['orientation_hold_exists'].notnull(), 'n', 'w'))
        df['verified'] = np.where(df['ver_complete'].notnull(), 'y', np.where(df['selected_for_ver'].notnull(), 'n', 'w'))
        df['sat10_total_score'] = (36-9) / (1600-590) * (df['sat10_total_score']-590) + 9
        df['act_equiv'] = df[['act_new_comp_score','sat10_total_score']].max(axis=1)
        for k in ['reading', 'writing', 'math']:
            df[k] = ~df[k].isin(['not college ready', 'retest required', pd.NA])
        return write(fn, df.drop(columns=self.flg_col['temp']+['cycle_day'], errors='ignore').dropna(axis=1, how='all').binarize())


    def get_raw(self):
        self['reg'] = {k: self.get_reg(cycle_day) for k, cycle_day in {'end':0, 'cur':self.cycle_day}.items()}
        # self['reg'] = {k: self.get_reg(cycle_day) for cycle_day in [0,self.cycle_day]]
        
        nm = 'raw'
        cycle_day, fn, df = self.prep(nm, self.cycle_day)
        if df is None:
            self.adm = self.get_adm(cycle_day)
            self.flg = self.get_flg(cycle_day)
            self.dst = self.get_dst(cycle_day)
            df =  (
                self.adm
                .merge(self.flg, how='left', on=['id','term_code_entry','styp_code'], suffixes=['_x',''])
                .merge(self.dst, how='left', on=['zip','camp_code'])
            )
            assert (df.groupby(['pidm','term_code']).size() == 1).all()
            write(fn, df)
        self[nm] = df
        return self