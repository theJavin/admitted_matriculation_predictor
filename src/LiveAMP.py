from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn import set_config
set_config(transform_output="pandas")
code_desc = lambda x: [x+'_code', x+'_desc']

crse_codes = [
    '_anycrse',
    'agec2317',
    'ansc1119',
    'ansc1319',
    'anth2302',
    'anth2351',
    'arts1301',
    'arts1303',
    'arts1304',
    'arts3331',
    'biol1305',
    'biol1406',
    'biol1407',
    'biol2401',
    'biol2402',
    'busi1301',
    'busi1307',
    'chem1111',
    'chem1112',
    'chem1302',
    'chem1311',
    'chem1312',
    'chem1407',
    'chem1411',
    'chem1412',
    'comm1311',
    'comm1315',
    'comm2302',
    'crij1301',
    'dram1310',
    'dram2361',
    'dram4304',
    'easc2310',
    'econ1301',
    'econ2301',
    'engl1301',
    'engl1302',
    'engl2307',
    'engl2320',
    'engl2321',
    'engl2326',
    'engl2340',
    'engl2350',
    'engl2360',
    'engl2362',
    'engl2364',
    'engl2366',
    'engl2368',
    'engr1211',
    'engr2303',
    'envs1302',
    'fina1360',
    'geog1303',
    'geog1320',
    'geog1451',
    'geog2301',
    'geol1403',
    'geol1404',
    'geol1407',
    'geol1408',
    'govt2305',
    'govt2306',
    'hist1301',
    'hist1302',
    'hist2321',
    'hist2322',
    'huma1315',
    'kine1301',
    'kine1338',
    'kine2315',
    'math1314',
    'math1316',
    'math1324',
    'math1325',
    'math1332',
    'math1342',
    'math2412',
    'math2413',
    'musi1303',
    'musi1310',
    'musi1311',
    'musi2350',
    'musi3325',
    'phil1301',
    'phil1304',
    'phil2303',
    'phil3301',
    'phys1302',
    'phys1401',
    'phys1402',
    'phys1403',
    'phys1410',
    'phys1411',
    'phys2425',
    'phys2426',
    'psyc2301',
    'psyc3303',
    'psyc3307',
    'soci1301',
    'soci1306',
    'soci2303',
    'univ0200',
    'univ0204',
    'univ0301',
    'univ0314',
    'univ0324',
    'univ0332',
    'univ0342',
]

@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day : int = 0
    proj_code : int = 202408
    term_codes: tuple = (202108, 202208, 202308, 202408)
    crse_code : str = '_anycrse'
    styp_code : str = 'n'
    stats: tuple = (pctl(0), pctl(25), pctl(50), pctl(75), pctl(100))
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = f"/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources/rslt"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y'})
    aggregations: tuple = (
        'crse_code',
        'coll_desc',
        'dept_desc',
        'majr_desc',
        'camp_desc',
        'gender',
        *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
        'waiver',
        'hs_qrtl',
        'international',
        'resd_desc',
        'lgcy',
        'lgcy_desc',
        'math',
        'reading',
        'writing',
        'oriented',
    )

    def __post_init__(self):
        super().__post_init__()
        self.root_path /= rjust(self.cycle_day,3,0)
        self.hash_str = join([str(v[0]) for v in self.param.values()], '/')


    def get_terms(self):
        def func():
            print()
            self.terms = {key: {term_code:
                Term(term_code=term_code, cycle_day=cycle_day, overwrite=self.overwrite, show=self.show).get_reg().get_raw()
            for term_code in self.term_codes} for key, cycle_day in {'current':self.cycle_day, 'actual':0}.items()}
            del self.terms['actual'][self.proj_code]
            self.cycle_date = self.terms['current'][self.proj_code].cycle_date
        return self.get(func, fn="terms.pkl")


    def get_X(self):
        def func():
            def get_raw(dct):
                R = pd.concat([T.raw for term_code, T in dct.items()]).dropna(axis=1, how='all').reset_index(drop=True).prep().copy()
                repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
                R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
                R['remote'] = R['camp_code'] != 's'
                R['resd'] = R['resd_code'] == 'r'
                R['oriented'] = R['oriented'] != 'n'
                R['lgcy'] = ~R['lgcy_code'].isin(['o',pd.NA])
                R['majr_code'] = R['majr_code'].replace({'0000':pd.NA, 'und':pd.NA, 'eled':'eted', 'agri':'unda'})
                R['coll_code'] = R['coll_code'].replace({'ae':'an', 'eh':'ed', 'hs':'hl', 'st':'sm', '00':pd.NA})
                R['coll_desc'] = R['coll_code'].map({
                    'an': 'ag & natural_resources',
                    'ba': 'business',
                    'ed': 'education',
                    'en': 'engineering',
                    'hl': 'health sciences',
                    'la': 'liberal & fine arts',
                    'sm': 'science & mathematics',
                    pd.NA: 'no college designated',
                })
                majr = ['majr_desc','dept_code','dept_desc','coll_code','coll_desc']
                S = R.sort_values('cycle_date').drop_duplicates(subset='majr_code', keep='last')[['majr_code',*majr]]
                X = R.drop(columns=majr).merge(S, on='majr_code', how='left').prep(bool=True)
                checks = [
                    'cycle_day >= 0',
                    'apdc_day >= cycle_day',
                    'appl_day >= apdc_day',
                    'birth_day >= appl_day',
                    'birth_day >= 5000',
                    'distance >= 0',
                    'hs_pctl >=0',
                    'hs_pctl <= 100',
                    'hs_qrtl >= 0',
                    'hs_qrtl <= 4',
                    'act_equiv >= 1',
                    'act_equiv <= 36',
                    'gap_score >= 0',
                    'gap_score <= 100',
                ]
                for check in checks:
                    mask = X.eval(check)
                    assert mask.all(), [check,X[~mask].disp(5)]
                return X

            idx = ['term_code','pidm']
            with warnings.catch_warnings(action='ignore'):
                X = {key: get_raw(dct).set_index(idx) for key, dct in self.terms.items()}
            Y = {key:
                pd.concat([T.reg for term_code, T in dct.items()])
                .dropna().sort_values(['crse_code',*idx]).set_index(idx).astype('string')
                # .assign(credit_hr=lambda y: y.eval("crse_code=='_allcrse'")*y['credit_hr'] + y.eval("crse_code!='_allcrse'")*y['crse_code'])
                .assign(credit_hr=lambda y: np.where(y.eval("crse_code=='_allcrse'"), y['credit_hr'], y['crse_code']))
                for key, dct in self.terms.items()}

            idx = ['variable','term_code','pidm']
            attr = ['id','levl_code','styp_code','admit','enroll','matric']
            crse = {'crse_code':'variable','credit_hr':'value'}
            def g(key):
                g = lambda variable, df: df[[]].assign(variable=variable, value=True)
                h = lambda D: D[key].drop(columns=crse, errors='ignore').melt(ignore_index=False)
                Z = pd.concat([
                    g('admit' , X['current']),
                    g('enroll', Y['current'].query("crse_code=='_anycrse'")),
                    g('matric', Y['actual' ].query("crse_code=='_anycrse'")),
                    Y[key].filter(crse).rename(columns=crse),
                    h(Y),
                    h(X),
                ]).dropna().astype('string').groupby(idx, sort=False).first()
                mask = Z.eval("variable in @attr")
                Z = Z[mask].unstack(0).droplevel(0,1).prep(bool=True).join(Z[~mask]).query("levl_code=='ug' & styp_code in ('n','r','t')").reset_index()
                Z['pidm'] = encrypt(Z['pidm'])
                Z['id'] = encrypt(Z['id'])
                Z.loc[Z.eval("variable==value"), "variable"] = "crse_code"
                return Z.set_index(idx+attr)
            self.Z = {'current':g('current').query('admit'), 'actual':g('actual').query('matric')}
            g = lambda df: df.groupby(['variable','term_code','levl_code','styp_code','value']).size()
            dct = {
                'admit' : g(self.Z['current']),
                'enroll': g(self.Z['current'].query('enroll')),
                'actual': g(self.Z['actual']),
            }
            dct['mlt'] = dct['actual'] / g(self.Z['actual'].query('admit'))
            A = pd.DataFrame(dct)
            mask = A.eval(f"term_code!={self.proj_code}")
            B = A[mask]
            C = A[~mask].drop(columns='mlt').join(B['mlt'].rename(index={k:self.proj_code for k in self.term_codes}))
            self.agg = pd.concat([B,C])
            self.y_true = pd.concat([z.loc['crse_code'].rsindex(['value',*idx]).assign(**{key:True}) for key,z in self.Z.items()], axis=1).prep(bool=True).fillna(False)
            X = self.Z['current'].query("variable!='crse_code'").unstack(0).droplevel(0,1).prep(bool=True)
            fill = {
                '_allcrse': 0,
                'birth_day': ['median',['term_code','styp_code']],
                'remote': False,
                'international': False,
                **{f'race_{r}': False for r in ['american_indian','asian','black','pacific','white','hispanic']},
                'lgcy': False,
                'resd': False,
                'waiver': False,
                'fafsa_app': False,
                'schlship_app': False,
                'finaid_accepted': False,
                'ssb': False,
                'math': False,
                'reading': False,
                'writing': False,
                'gap_score': 0,
                'oriented': False,
            }
            for key, val in fill.items():
                X[key] = X.impute(key, *listify(val))
            M = X.isnull().rename(columns=lambda x:x+'_missing')
            self.X = X.join(M).prep(bool=True)
        return self.get(func, pre="terms", drop="terms", fn="X.pkl")


    def get_X_proc(self):
        def func():
            X = self.X.query(f"styp_code==@self.styp_code").rename(columns=lambda x:x+'_')
            trf = ColumnTransformer(self.param['trf'][2], remainder='drop', verbose_feature_names_out=False)
            X_trf = trf.fit_transform(X).prep(bool=True, cat=True)
            imp_dct = self.param['imp'][2].copy()
            iterations = imp_dct.pop('iterations')
            tune = imp_dct.pop('tune')
            if tune:
                ds = imp_dct.pop('datasets')
                imp = mf.ImputationKernel(self.X_trf, datasets=1, **imp_dct)
                imp.mice(iterations)
                variable_parameters, losses = imp.tune_parameters(dataset=0)
                imp_dct['datasets'] = ds
            else:
                variable_parameters = None
            imp = mf.ImputationKernel(X_trf, **imp_dct)
            imp.mice(iterations, variable_parameters=variable_parameters)
            self.X_proc = pd.concat([
                    imp.complete_data(k)
                    # .addlevel({'sim':k, 'trf_hash':self.param['trf'][0], 'imp_hash':self.param['imp'][0]})
                    .addlevel({'sim':k})
                    .prep(bool=True, cat=True)
                for k in range(imp.dataset_count())])
        return self.get(func, pre="X", fn=f"X_proc/{self.hash_str.rsplit('/',1)[0]}/{self.styp_code}.pkl")


    def get_y_pred(self):
        def func():
            Z = (
                self.X_proc
                .join(self.y_true.loc[self.crse_code])
                .fillna({c:False for c in self.y_true.columns})
                .sort_values(['actual','act_equiv_missing_','pidm'], ascending=False)
                .assign(mask=lambda x: x.groupby(['term_code','sim']).cumcount()%5==0)
            )
            X = Z.drop(columns=['actual','mask']).copy()
            Z_model = Z.groupby(['term_code','sim']).filter(lambda z: z['actual'].sum()>=5)
            self.clf = dict()
            self.y_pred = dict()
            self.train_score = dict()
            for train_code in self.term_codes:
                print(train_code, end="...")
                X_model = Z_model.query("term_code==@train_code" if train_code != self.proj_code else "term_code!=@train_code").copy()
                y_model = X_model.pop('actual')
                mask = X_model.pop('mask')
                if len(X_model) > 0:
                    dct = self.param['clf'][2] | {
                        'X_train':X_model[~mask],
                        'y_train':y_model[~mask],
                        'X_val':X_model[mask],
                        'y_val':y_model[mask],
                        'task':'classification',
                        'verbose':0,
                    }
                    clf = fl.AutoML(**dct)
                    with warnings.catch_warnings(action='ignore'):
                        clf.fit(**dct)
                    predict = clf.predict(X)
                    proba = clf.predict_proba(X)[:,1]
                    self.train_score[train_code] = clf.best_result['val_loss'] * 100
                    self.clf[train_code] = clf._trained_estimator
                    print('done', end="  ")
                else:
                    print('fail', end="  ")
                    predict = pd.NA
                    proba = pd.NA
                    self.train_score[train_code] = pd.NA
                self.y_pred[train_code] = Z.assign(predict=predict,proba=proba).prep(bool=True).reset_index()[['pidm','term_code','sim','actual','predict','proba']]
        return self.get(func, pre="X_proc", drop=["terms","X","y_true","Z","agg","X_trf","X_proc"],
                        fn=f"y_pred/{self.hash_str}/{self.crse_code}/{self.styp_code}.pkl")


    def get_stack(self, crse_code="_anycrse"):
        L = []
        for fn in (self.root_path / "y_pred").rglob("*.pkl"):
            if all(x in str(fn) for x in [self.hash_str, crse_code]):
                dct = read(fn)
                for train_code, y_pred in dct['y_pred'].items():
                    y = y_pred.assign(crse_code=dct['crse_code'], train_code=train_code, train_score=dct['train_score'][train_code])
                    L.append(y)
                del dct
        return pd.concat(L).prep(bool=True)


    def get_result(self, variable):
        del self['result']
        def func():
            grp = [variable,'levl_code','styp_code','term_code','train_code','sim'] #,'trf_hash','imp_hash','clf_hash']
            Y = self.get_stack("" if variable=="crse_code" else "_anycrse")
            X = self.X[variable if variable!="crse_code" else []].reset_index()
            A = self.agg.loc[variable].reset_index().rename(columns={'value':variable}).prep(bool=True)
            S = (Y.merge(X,'left').groupby(grp).apply(lambda y: pd.Series({
                    'predict': y['proba'].sum(),
                    'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
                    'train_score': y['train_score'].mean(),
                    # 'train_score_var': y['train_score'].var(),
                }), include_groups=False)
                # .join(A).prep().copy()
                .reset_index().merge(A,'left').set_index(grp).copy()
            )
            S['predict'] *= S['mlt']
            mask = lambda key: S[key].isnull() | (S[key]==0)
            S.loc[mask('predict'), ['test_score','train_score','predict','mlt']] = pd.NA
            S.loc[mask('actual'),  ['test_score']] = pd.NA
            P = S['actual'].rename('prior').reset_index().copy()
            P['term_code'] += 100
            S = S.reset_index().merge(P,'left')
            alpha = 1
            S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
            S['error'] = S['predict'] - S['actual']
            S['error_pct'] = S['error'] / S['actual'] * 100
            S['change'] = S['predict'] - S['prior']
            S['change_pct'] = S['change'] / S['prior'] * 100
            S = (S
                .prep()
                .sort_values(grp, ascending=[True,True,True,False,False,True])
                .set_index(grp)
                [['admit','enroll','actual','prior','predict','change','change_pct','error','error_pct','train_score','test_score','overall_score','mlt']]
            )
            grp.remove('sim')
            self.result = {'summary':S} | {str(stat):S.groupby(grp,sort=False).agg(stat).prep() for stat in listify(self.stats)}
            # self.result[' 50%'].disp(100)
        return self.get(func, pre="X", drop=["terms","X","y_true","Z","agg","X_trf","X_proc","y_pred","clf"], fn=f"result/{self.hash_str}/{variable}.pkl")


    def get_report(self):
        path = str(self.root_path / f"result/{self.hash_str}/AMP_{self.cycle_date.date()}")
        A = listify(self.aggregations)
        self.get_result(A[0])
        with pd.ExcelWriter(path+".xlsx", mode="w", engine="openpyxl") as writer:
            df = pd.DataFrame({'':[
                f"Admitted Matriculation Projections (AMP) for {self.cycle_date.date()}",
                f"AMP is a machine learning algorithm (XGBoost) to forecast the number, characteristics, and likely course enrollments of incoming Fall cohort based on their application information and status in pre-semester processes (orientation, course registration, financial aid, etc).",
                f"Broadly, for each student admitted for the upcoming fall (2024), AMP identifies similar students from prior fall cohorts (2021-23) and traces which courses they eventually took (if any).",
                f"More precisely, for each (incoming student, course) pair, AMP assigns a probability that the student will take that course based on students from prior years.",
                f"These student-level predictions are aggregated/filtered at many levels (course, department, college, race/ethnicity, gender, origin, preparedness, etc) on different sheets in this workbook.",
                f"AMP serves primarily as a planning tool so leaders (deans, dept heads, housing, etc) can proactively allocate appropriate resources (instructors, sections, labs, etc) in advance.",
                f"AMP predictions should be used with caution in combination with the knowledge and expertise of human leaders - it is far from perfect."
                f"",
                f"AMP is based on EM's Flags report & IDA's daily admissions/registration snapshots.",
                f"AMP learns from PAST experiences captured in these 3 data souces only. It does not know about recent changes or anything not captured in these 3 limited datasets."
                f"EM's Flags report pulls from admission applications and may NOT reflect changes made later (ex: change major/dept/college).",
                f"",
                f"term_desc = term to be predicted,   train_desc = term(s) to be trained on",
                f"MOST USERS ONLY NEED SUMMARY ON LEFT-HAND SIDE OF ANY SHEET which reflects term_desc=2024fall (prediction for forthcoming term) and train_desc=all (model trained on all available prior years: 21fl, 22fl, 23fl)",
                f"Some power users might value detailed results on right-hand side",
                f"",
                f"admit = # admitted on this day of term_desc",
                f"enroll = # enrolled on this day of term_desc",
                f"actual = # enrolled on census day of term_desc",
                f"prior = # enrolled on census day 1 year prior to term_desc",
                f"predict = # AMP predicts will be enrolled on census day if term_desc",
                f"change = predict - prior",
                f"change_pct = change / prior * 100",
                f"error = predict - actual",
                f"error_pct = error / actual * 100",
                f"train_score = log_loss metric during model training",
                f"test_score = log_loss metric during model validation",
                f"overall_score = mean of train & test scores",
                f"mlt = future applications multiplier",
                f"",
                f"DETAILS",
                f"AMP predicts only for courses with enrollment >=5",
                ]}).to_excel(writer, sheet_name='instructions', index=False)

            for variable in A:
                R = self.get_result(variable).result[' 50%']
                rnd = ['admit','enroll','actual','prior','predict','change','error'] 
                R[rnd] = R[rnd].round()
                R = (R
                    .reset_index()
                    .assign(levl_desc =lambda r: r['levl_code' ].replace({'ug':'undergrad','g':'grad'}))
                    .assign(styp_desc =lambda r: r['styp_code' ].replace({'n':'new first time','t':'transfer','r':'returning'}))
                    .assign(term_desc =lambda r: r['term_code' ].astype('string').str.replace('08','Fall'))
                    .assign(train_desc=lambda r: r['train_code'].astype('string').replace(str(self.proj_code),'all').str.replace('08','Fall'))
                    .round(2)
                    .prep()
                )
                col = [variable,'levl_desc','styp_desc','term_desc','train_desc','admit','enroll','actual','prior','predict','change','change_pct','error','error_pct','overall_score','test_score','train_score','mlt']
                S = R.query(f"term_code=={self.proj_code} & train_desc=='all'")[col].drop(columns=['term_desc','train_desc','admit','actual','error','error_pct','overall_score','test_score','train_score','mlt'])
                S.to_excel(writer, sheet_name=variable, index=False)
                R[col].to_excel(writer, sheet_name=variable, index=False, startcol=S.shape[1]+4)
                format_xlsx(writer.sheets[variable], freeze_cols=3)


        self.get_X()
        cols = [x for x in self.X.columns if '_missing' not in x]
        X = self.X.reset_index(['admit','enroll','matric'])[[*cols,'enroll']]
        Y = self.get_stack().query(f"train_code=={self.proj_code}").groupby(['pidm','term_code'])['proba'].agg(probability_mean='mean',probability_std='std') * 100
        Z = X.join(Y).reset_index()
        Z['pidm'] = decrypt(Z['pidm'])
        Z['id'] = decrypt(Z['id'])
        # with pd.ExcelWriter(path+"_details.xlsx", mode="w", engine="xlsxwriter") as writer:
        with pd.ExcelWriter(path+"_details.xlsx", mode="w", engine="openpyxl") as writer:
            variable = 'probability'
            Z.sort_values(['term_code','pidm'], ascending=[False,True]).to_excel(writer, sheet_name=variable, index=False)
            format_xlsx(writer.sheets[variable], freeze_cols=5)

        print('DONE!')
        return self

pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())
param_grds = {
    'trf': {
        '_allcrse': 'passthrough',
        'act_equiv': 'passthrough',
        'act_equiv_missing': 'passthrough',
        'admt_code': 'drop',
        'apdc_day': 'passthrough',
        'appl_day': 'drop',
        'birth_day': 'passthrough',
        'camp_code': 'drop',
        'coll_code': 'passthrough',
        # 'distance': ['passthrough', pwrtrf],
        'distance': 'passthrough',
        'fafsa_app': 'drop',
        'finaid_accepted': 'drop',
        'gap_score': 'passthrough',
        'gender': 'passthrough',
        'hs_qrtl': 'passthrough',
        'international': 'passthrough',
        'lgcy': 'passthrough',
        'math': 'passthrough',
        'oriented': 'passthrough',
        'term_code': 'drop',
        **{f'race_{r}': 'passthrough' for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'reading': 'passthrough',
        'remote': 'passthrough',
        'resd': 'passthrough',
        'schlship_app': 'passthrough',
        'ssb': 'passthrough',
        'waiver': 'passthrough',
        'writing': 'passthrough',
    },
    'imp': {
        'random_state': seed,
        'datasets': 10,
        'iterations': 10,
        # 'datasets': 3,
        # 'iterations': 2,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'seed': seed,
        'metric': 'log_loss',
        'early_stop': True,
        # 'time_budget': 1,
        'time_budget': 120,
        'estimator_list': [['xgboost']],
        'ensemble': False,
        # 'ensemble': [False, True],
    },
}


formatter = lambda x: str(x).replace('\n','').replace(' ','')
hasher = lambda x, d=2: hashlib.shake_256(formatter(x).encode()).hexdigest(d)
# hasher = lambda x, d=2: int.from_bytes(hashlib.shake_256(formatter(x).encode()).digest(d))
param_dct = dict()
for key, val in param_grds.items():
    lst = cartesian(val, sort=True, key=str)
    if key == 'trf':
        lst = [[(c,t,[c+'_']) for c,t in trf.items() if t not in ['drop', None, '']] for trf in lst]
    param_dct[key] = [[hasher(k), formatter(k), k] for k in lst]
param_lst = cartesian(param_dct)


def run_amp(cycle_day, *styp_codes):
    cycle_day = int(cycle_day)
    styp_codes = styp_codes if styp_codes else ['n','r','t']
    self = AMP(cycle_day=cycle_day).get_X()
    for kwargs in cartesian({'crse_code':intersection(crse_codes, self.y_true.reset_index()['value'], sort=True, reverse=True), 'cycle_day':cycle_day, 'styp_code':styp_codes, 'param':param_lst}):
        self = AMP(**kwargs).get_y_pred()
    self.get_report()
    return self


if __name__ == "__main__":
    delattr(pd.Series, 'disp')
    delattr(pd.DataFrame, 'disp')
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        print()
        print(df.reset_index().drop(columns='index', errors='ignore').head(max_rows).to_markdown(tablefmt='psql'))

    sys.stdout = sys.stderr  # unbuffer output - emulates -u
    print(pd.Timestamp.now())
    run_amp(*sys.argv[1:])