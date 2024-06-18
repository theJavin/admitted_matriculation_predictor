from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
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

    'psyc3303',
    'psyc3307',
    'kine1301',
    'kine1338',
    'musi1310',
    'musi1303',
    'engr1211',
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
        # 'dept_desc',
        # 'majr_desc',
        # 'camp_desc',
        # # # # 'stat_desc',
        # # # # 'cnty_desc',
        # 'gender',
        # *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
        # 'waiver',
        # 'hs_qrtl',
        # 'international',
        # 'resd_desc',
        'lgcy',
        # 'lgcy_desc',
        # # # # 'admt_desc',
        # 'math',
        # 'reading',
        # 'writing',
        # # # # 'ssb',
        # 'oriented',
    )

    def __post_init__(self):
        super().__post_init__()
        self.root_path /= rjust(self.cycle_day,3,0)
        self.hash_path = join([str(v[0]) for v in self.param.values()], '/')


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
        return self.get(func, pre="X", fn=f"X_proc/{self.hash_path.rsplit('/',1)[0]}/{self.styp_code}.pkl")


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
            # self.clf = dict()
            self.y_pred = dict()
            self.train_score = dict()
            for train_code in self.term_codes:
            # for train_code in [self.proj_code]:
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
                    # self.clf[train_code] = clf._trained_estimator
                    print('done', end="  ")
                else:
                    print('fail', end="  ")
                    predict = pd.NA
                    proba = pd.NA
                    self.train_score[train_code] = pd.NA
                # self.y_pred[train_code] = (Z
                #     .assign(predict=predict,proba=proba)
                #     # .astype({'predict':'float', 'proba':'boolean'})
                #     .reset_index()
                #     [['pidm','term_code','sim','actual','predict','proba']]
                #     .prep(bool=True))
                self.y_pred[train_code] = Z.assign(predict=predict,proba=proba).prep(bool=True).reset_index()[['pidm','term_code','sim','actual','predict','proba']]
                # self.y_pred[train_code] = Z.assign(predict=predict,proba=proba).astype({'predict':'boolean', 'proba':'float', 'train_score':'float'}).prep(bool=True)[['pidm','term_code','sim','actual','predict','proba']].prep(bool=True)
                # y[train_code] = Z.rsindex(['term_code','pidm'])['actual'].assign(pred=pred, proba=proba, train_score=train_score, train_code=train_code)
                # y[train_code] = Z.reset_index()[['term_code','pidm','actual']].assign(pred=pred, proba=proba, train_code=train_code)
                # y[train_code] = Z[['actual']].assign(pred=pred, proba=proba, train_score=score, train_code=train_code, crse_code=self.crse_code, clf_hash=self.param['clf'][0])#.prep(bool=True)
                # y[train_code] = Z[['actual']].assign(pred=pred, proba=proba, train_code=train_code, crse_code=self.crse_code, clf_hash=self.param['clf'][0]).prep(bool=True)
                # train_score[train_code] = score
            # self.y_pred = pd.concat(y.values()).prep(bool=True)
            # self.y_pred = pd.concat(y.values()).prep(bool=True).rsindex(['crse_code','pidm','term_code','train_code','sim','trf_hash','imp_hash','clf_hash'])
            # self.y_pred = pd.concat(y.values()).prep(bool=True).rsindex(['crse_code','pidm','term_code','train_code','sim','trf_hash','imp_hash','clf_hash'])
            # print(self.y_pred.shape)
            # self.y_pred.dtypes.disp(100)
            
            # y = pd.concat(y.values()).prep(bool=True).reset_index()#.rsindex(['crse_code','pidm','id','levl_code','styp_code','term_code','train_code','sim','trf_hash','imp_hash','clf_hash'])
            # self.train_score = y.groupby(['crse_code','levl_code','styp_code','train_code','trf_hash','imp_hash','clf_hash'])['train_score'].mean().reset_index()
            # self.y_pred = y[['crse_code','pidm','term_code','train_code','sim','trf_hash','imp_hash','clf_hash','actual','pred','proba']].copy()
            # self.y_pred = y.drop(columns=['id','levl_code','styp_code','train_score'])

            # self.train_score = y.pop('train_score').groupby(['crse_code','levl_code','styp_code','train_code','trf_hash','imp_hash','clf_hash']).mean()
            # self.y_pred = y.droplevel(['id','levl_code','styp_code'])
            
            # self.train_score.disp(100)
            # assert 1==2
            
            # self.train_score = y.pop('train_score').droplevel(['pidm','id','term_code','sim'])
            
            
            # droplevel(['pidm','id',])
            # y.disp()
            # y.rsindex(['crse_code','pidm','id','levl_code','styp_code','term_code','train_code','sim','trf_hash','imp_hash','clf_hash']).disp()
            # assert 1==2
                                                               
            # #                                                    term_code','train_code','sim','trf_hash','imp_hash','clf_hash']
            # # s = y.pop
            
            
            # self.y_pred = pd.concat(y.values())
            # self.y_pred.disp()
            # # .rsindex(['crse_code','pidm','term_code','train_code','sim','trf_hash','imp_hash','clf_hash'])
            # self.y_pred.pop('train_score').groupby(['crse_code','train_code','trf_hash','imp_hash','clf_hash']).var().disp(100)
            # assert 1==2
            # .droplevel(['pidm'])#,'term_code','sim'])
            # self.train_score = self.y_pred.pop('train_score').droplevel(['pidm'])#,'term_code','sim'])
            # self.train_score.disp
            # self.train_score = pd.Series(train_score, name='train_score').rename_axis('train_code').addlevel({
            #     'crse_code':self.crse_code, 'levl_code':'ug', 'styp_code':self.styp_code})
        return self.get(func, pre="X_proc", drop=["terms","X","y_true","Z","agg","X_trf","X_proc"],
                        # fn=f"y_pred/{self.crse_code}/{self.styp_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl")
                        # fn=f"y_pred/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}/{self.crse_code}/{self.styp_code}.pkl")
                        fn=f"y_pred/{self.hash_path}/{self.crse_code}/{self.styp_code}.pkl")


    # def get_y_stack(self):
    #     def func():
    #         L = []
    #         for fn in sorted((self.root_path / 'y_pred').rglob('*.pkl')):
    #             # print(fn)
    #             dct = read(fn)
    #             for train_code, y_pred in dct['y_pred'].items():
    #                 y = (y_pred.assign(
    #                         crse_code=dct['crse_code'],
    #                         train_code=train_code,
    #                         train_score=dct['train_score'][train_code],
    #                         **{key+'_hash': val[0] for key, val in dct['param'].items()})
    #                     # .astype({'predict':'float', 'proba':'boolean', 'train_score':'float'})
    #                     # .prep(bool=True)
    #                 )
    #                 # y = y.assign(crse_code=dct['crse_code'], train_code=train_code, train_score=dct['train_score'][train_code], **{key+'_hash': val[0] for key, val in dct['param'].items()}).prep()
    #                 # y.dtypes.disp(100)
    #                 L.append(y)
    #             del dct
    #         self.y_stack = pd.concat(L).prep(bool=True)
    #         # self.y_stack = pd.concat(Y).astype({'predict':'boolean', 'proba':'float', 'train_score':'float'}).prep(bool=True)
    #     return self.get(func, pre="y_pred", fn=f"y_stack.pkl", subpath="result", drop=["terms","X","y_true","Z","agg","X_trf","X_proc","y_pred","clf"])


    def get_result(self, variable):
        del self['result']
        def func():
            # grp = [variable,'levl_code','styp_code','term_code','train_code','sim','trf_hash','imp_hash','clf_hash']
            grp = [variable,'levl_code','styp_code','term_code','train_code','sim']
            L = []
            for fn in (self.root_path / "y_pred").rglob("*.pkl"):
                if all(str(x) in str(fn) for x in [v[0] for v in self.param.values()]+["" if variable=="crse_code" else "_anycrse"]):
                    dct = read(fn)
                    for train_code, y_pred in dct['y_pred'].items():
                        y = y_pred.assign(crse_code=dct['crse_code'], train_code=train_code, train_score=dct['train_score'][train_code])
                        L.append(y)
                    del dct
            Y = pd.concat(L).prep(bool=True)
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
                [['admit','enroll','actual','prior','predict','change','change_pct','error','error_pct','overall_score','test_score','train_score','mlt']]
            )
            grp.remove('sim')
            self.result = {'summary':S} | {str(stat):S.groupby(grp,sort=False).agg(stat).prep() for stat in listify(self.stats)}
            # self.result[' 50%'].disp(100)
        return self.get(func, pre="X", drop=["terms","X","y_true","Z","agg","X_trf","X_proc","y_pred","clf"], fn=f"result/{self.hash_path}/{variable}.pkl")


    def get_report(self):
        from openpyxl.styles import Alignment
        from openpyxl.utils import get_column_letter
        A = listify(self.aggregations)
        self.get_result(A[0])
        with pd.ExcelWriter(self.root_path / f"result/{self.hash_path}/AMP_{self.cycle_date.date()}.xlsx", mode="w", engine="openpyxl") as writer:
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
                grp = [variable,'levl_desc','styp_desc']
                feat = ['enroll','prior','predict','change','change_pct']
                S = R.query(f"term_code=={self.proj_code} & train_desc=='all'")[grp+feat]
                S.to_excel(writer, sheet_name=variable, index=False)
                
        #         # S = R.query(f"term_code=={self.proj_code} & train_desc=='all'")[[variable,'styp_desc','enroll','prior','predict','change','change_pct']]
        #         grp = [*grp,'term_desc','train_desc']
        #         feat = ['admit',*feat,'error','error_pct','overall_score','test_score','train_score','mlt']
        #         R[grp+feat].to_excel(writer, sheet_name=variable, index=False, startcol=S.shape[1]+2)#, startrow=0)
        #         sheet = writer.sheets[variable]
        #         sheet.freeze_panes = "A2"
        #         sheet.auto_filter.ref = sheet.dimensions
        #         for k, column in enumerate(sheet.columns):
        #             width = 3+max(len(str(cell.value)) for cell in column)
        #             sheet.column_dimensions[get_column_letter(k+1)].width = width
        #         for cell in sheet[1]:
        #             cell.alignment = Alignment(horizontal="left")

        # print('DONE!')
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
        # 'datasets': 10,
        # 'iterations': 10,
        'datasets': 3,
        'iterations': 2,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'seed': seed,
        'metric': 'log_loss',
        'early_stop': True,
        'time_budget': 1,
        # 'time_budget': 120,
        'estimator_list': [['xgboost']],
        'ensemble': False,
        # 'ensemble': [False, True],
    },
}


formatter = lambda x: str(x).replace('\n','').replace(' ','')
# hasher = lambda x, d=2: hashlib.shake_128(formatter(x).encode()).hexdigest(d)
hasher = lambda x, d=2: int.from_bytes(hashlib.shake_256(formatter(x).encode()).digest(d))
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
    # self.get_y_stack()
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