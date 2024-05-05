from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, fbeta_score, f1_score
from sklearn import set_config
set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())

################## Metrics that work for FLAML and sklearn ##################
class Metric(MyBaseClass):
    """Parent class for metrics that work with FLAML and sklearn"""
    def __str__(self):
        return self.__repr__()
    def __lt__(self, other):
        return str(self) < str(other)
    def __call__(self, X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, *args):
        """for FLAML"""
        start = time.time()
        y_pred = estimator.predict(X_val)
        pred_time = (time.time() - start) / len(X_val)
        val_loss = self.score(y_val, y_pred, labels=labels, sample_weight=weight_val)
        y_pred = estimator.predict(X_train)
        train_loss = self.score(y_train, y_pred, labels=labels, sample_weight=weight_train)
        return val_loss, {
            "val_loss": val_loss,
            "train_loss": train_loss,
            "pred_time": pred_time,
        }

class F_beta(Metric):
    """Implements F_beta for FLAML https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html"""
    def __init__(self, beta):
        self.beta = float(beta)
    def __repr__(self):
        return f"F_{self.beta:.2f}"
    def score(self, y_true, y_pred, **kwargs):
        return 1 - fbeta_score(y_true, y_pred, beta=self.beta, **kwargs)

class Accuracy(Metric):
    def __repr__(self):
        return "accuracy"
    def score(self, y_true, y_pred, **kwargs):
        return 1 - accuracy_score(y_true, y_pred, **kwargs)

class LogLoss(Metric):
    def __repr__(self):
        return "log loss"
    def score(self, y_true, y_pred, **kwargs):
        return log_loss(y_true, y_pred, **kwargs)

################## AMP ##################
crse_codes = [
    '_allcrse',
    # 'agec2317',
    # # 'ansc1119',
    # 'ansc1319',
    # # 'anth2302',
    # # 'anth2351',
    # 'arts1301',
    # 'arts1303',
    # 'arts1304',
    # 'arts3331',
    # 'biol1305',
    'biol1406',
    # 'biol1407',
    # 'biol2401',
    # 'biol2402',
    # 'busi1301',
    # 'busi1307',
    # 'chem1111',
    # 'chem1112',
    # 'chem1302',
    # 'chem1311',
    # 'chem1312',
    # 'chem1407',
    # 'chem1411',
    # 'chem1412',
    # 'comm1311',
    # 'comm1315',
    # 'comm2302',
    # 'crij1301',
    # 'dram1310',
    # 'dram2361',
    # 'dram4304',
    # 'easc2310',
    # 'econ1301',
    # 'econ2301',
    # 'engl1301',
    # 'engl1302',
    # 'engl2307',
    # 'engl2320',
    # 'engl2321',
    # 'engl2326',
    # 'engl2340',
    # 'engl2350',
    # 'engl2360',
    # 'engl2362',
    # 'engl2364',
    # 'engl2366',
    # 'engl2368',
    # 'engr2303',
    # 'envs1302',
    # 'fina1360',
    # 'geog1303',
    # 'geog1320',
    # 'geog1451',
    # 'geog2301',
    # 'geol1403',
    # 'geol1404',
    # 'geol1407',
    # 'geol1408',
    # 'govt2305',
    # 'govt2306',
    # 'hist1301',
    # 'hist1302',
    # 'hist2321',
    # 'hist2322',
    # 'huma1315',
    # 'kine2315',
    # 'math1314',
    # 'math1316',
    # 'math1324',
    # 'math1332',
    # 'math1342',
    # 'math2412',
    # 'math2413',
    # 'musi1303',
    # 'musi1310',
    # 'musi1311',
    # 'musi2350',
    # 'musi3325',
    # 'phil1301',
    # 'phil1304',
    # 'phil2303',
    # 'phil3301',
    # 'phys1302',
    # 'phys1401',
    # 'phys1402',
    # 'phys1403',
    # 'phys1410',
    # 'phys1411',
    # 'phys2425',
    # 'phys2426',
    # 'psyc2301',
    # 'soci1301',
    # 'soci1306',
    # 'soci2303',
    # 'univ0200',
    # 'univ0204',
    # 'univ0301',
    # 'univ0314',
    # 'univ0324',
    # 'univ0332',
    # 'univ0342',
]

@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day : int = None
    proj_code : int = 202408
    train_code: int = 202308
    pred_codes: tuple = (202108, 202208, 202308, 202408)
    crse_code : str = '_allcrse'
    styp_code : str = 'n'
    stats: tuple = (pctl(0), pctl(25), pctl(50), pctl(75), pctl(100))
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = f"/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources/rslt"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y'})

    def __post_init__(self):
        super().__post_init__()
        if self.cycle_day is None:
            self.cycle_day = (Term(term_code=202408).cycle_date-pd.Timestamp.now()).days+1
        self.root_path /= rjust(self.cycle_day,3,0)


    def get_X(self):
        def func():
            print()
            terms = {cycle_day: {pred_code:
                    Term(term_code=pred_code, cycle_day=cycle_day, overwrite=self.overwrite, show=self.show).get_reg().get_raw()
                for pred_code in self.pred_codes} for cycle_day in [0, self.cycle_day]}
            ren = {'term_code':'pred_code', 'term_desc':'pred_desc'}
            R = pd.concat([T.raw.rename(columns=ren) for pred_code, T in terms[self.cycle_day].items()]).copy().dropna(axis=1, how='all').reset_index(drop=True).prep()
            repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
            R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
            R['remote'] = R['camp_code'] != 's'
            R['resd'] = R['resd_code'] == 'r'
            R['oriented'] = R['oriented'] != 'n'
            R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
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

            fill = {
                'birth_day': ['median',['pred_code','styp_code']],
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
            for k, v in fill.items():
                X[k] = X.impute(k, *listify(v))
            M = X.isnull().rename(columns=lambda x:x+'_missing')

            code_desc = lambda x: [x+'_code', x+'_desc']
            attr = [
                'index',
                'pidm',
                *code_desc('pred'),
                *code_desc('apdc'),
                *code_desc('levl'),
                *code_desc('styp'),
                *code_desc('admt'),
                *code_desc('camp'),
                *code_desc('coll'),
                *code_desc('dept'),
                *code_desc('majr'),
                *code_desc('cnty'),
                *code_desc('stat'),
                *code_desc('natn'),
                *code_desc('resd'),
                *code_desc('lgcy'),
                'international',
                'gender',
                *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
                'waiver',
                'birth_day',
                'distance',
                'hs_qrtl',
            ]
            self.X = X.join(M).reset_index().set_index(attr, drop=False).rename(columns=lambda x:'__'+x).prep(bool=True)

            Y = pd.concat([
                    T.reg
                    .rename(columns=ren)
                    .assign(crse_code=lambda X: X['crse_code'] + ('_cur' if cycle_day > 0 else ''))
                    .set_index(['pidm','levl_code','styp_code','pred_code','crse_code'])
                    [['credit_hr']]
                    .fillna(0)
                for cycle_day, dct in terms.items() for pred_code, T in dct.items()]).copy().fillna(0).prep()
            Y.loc[Y.eval("crse_code!='_allcrse_cur' & credit_hr>0")] = 1
            self.Y = Y.rsindex(['pidm','pred_code','crse_code'])
            Z = self.X[[]].join(self.Y)#, how='inner')
            agg = lambda y: y.groupby(['styp_code','pred_code','crse_code']).sum().query(f"styp_code in ('n','r','t') and pred_code!={self.proj_code} and not crse_code.str.contains('cur')")
            self.mlt = agg(Y) / agg(Z)
        return self.get(func, "X.pkl")


    def get_X_proc(self):
        def func():
            trf = ColumnTransformer(self.param['trf'][2], remainder='drop', verbose_feature_names_out=False)
            X_trf = trf.fit_transform(self.X.query("styp_code==@self.styp_code")).prep(bool=True, cat=True).sample(frac=1)
            imp_dct = self.param['imp'][2].copy()
            iterations = imp_dct.pop('iterations')
            tune = imp_dct.pop('tune')
            if tune:
                ds = imp_dct.pop('datasets')
                imp = mf.ImputationKernel(X_trf, datasets=1, **imp_dct)
                imp.mice(iterations)
                variable_parameters, losses = imp.tune_parameters(dataset=0)
                imp_dct['datasets'] = ds
            else:
                variable_parameters = None
            imp = mf.ImputationKernel(X_trf, **imp_dct)
            imp.mice(iterations, variable_parameters=variable_parameters)
            self.X_proc = [
                imp.complete_data(k)
                .addlevels({'trf_hash':self.param['trf'][0], 'imp_hash':self.param['imp'][0], 'sim':k})
                .prep(bool=True, cat=True)
            for k in range(imp.dataset_count())]
            del self.X
        return self.get(func, f"X_proc/{self.styp_code}/{self.param['trf'][0]}/{self.param['imp'][0]}.pkl", "X")


    def get_Y(self):
        def func():
            cols = uniquify(['_allcrse_cur', self.crse_code+'_cur', self.crse_code])
            Y = self.Y.query(f"crse_code.isin({cols})").unstack().droplevel(0,1)
            if self.X_proc[0].query(f"pred_code=={self.train_code}").join(Y)[self.crse_code].sum() < 10:
                return 'fail'
            Z = pd.concat(self.X_proc).join(Y).fillna({c:0 for c in cols}).addlevels({'crse_code':self.crse_code, 'train_code':self.train_code, 'clf_hash':self.param['clf'][0]}).prep(bool=True, cat=True)
            # Z = (
            #     pd.concat([X.join(Y) for X in self.X_proc])
            #     .fillna({c:0 for c in cols})
            #     .addlevels({'crse_code':self.crse_code, 'train_code':self.train_code, 'clf_hash':self.param['clf'][0]})
            #     .prep(bool=True, cat=True)
            # )
            # clf_dct = self.param['clf'][2].copy() | {'split_type':'stratified', 'task':'classification', 'verbose':0, 'log_file_name': self.path.with_suffix('.log')}
            # n_calibrations = clf_dct.pop('n_calibrations')
            # clf_dct['log_file_name'].parent.mkdir(exist_ok=True, parents=True)

            # def train(Z, mask, weight=0, dct=clf_dct):
            #     clf = fl.AutoML(**dct)
            #     X = Z.copy()
            #     y = X.pop(self.crse_code)
            #     # clf.X = X
            #     # mask = X.eval(f"pred_code=={self.train_code}")
            #     with warnings.catch_warnings(action='ignore'):
            #         clf.fit(
            #             X[mask],
            #             y[mask],
            #             sample_weight=1+(2*y[mask]-1)*weight,
            #             **dct)
            #         clf.Y = y.rename('actual').to_frame().assign(
            #             proba=clf.predict_proba(X)[:,1],
            #             # predicted=clf.predict(clf.X),
            #             ).prep(bool=True).copy()
            #     return clf

            # print()
            # weight = 0
            # weight_hist = {weight: np.inf}
            # z = Z.query(f"pred_code=={self.train_code}").copy()
            # t = z.query(f"sim==0").groupby([self.crse_code,'__coll_code','__remote'], observed=True)
            # mask = z.reset_index()['index'].isin(t.sample(frac=0.7).reset_index()['index']).values
            # for i in range(n_calibrations):
            #     clf = train(z, mask, weight, clf_dct|{'time_budget':15})
            #     y = clf.Y[~mask].sum()
            #     y['err'] = y['proba'] / y['actual'] - 1
            #     y.to_frame().T.disp(10)
            #     weight_hist[weight] = abs(y['err'])
            #     print(weight, y['err'])
            #     if abs(y['err']) < 0.001:
            #         break
            #     # weight += (2*(np.mean(y['err'])<0)-1) / (2**(i+1))
            #     weight = np.clip(weight - y['err'], -1, 1)

                # y['err'] = np.where(y['actual']<10, pd.NA, y['proba'] / y['actual'] - 1)
                # y.disp(5)
                # mape = np.mean(np.abs(y['err'])) * 100
                # weight_hist[weight] = mape
                # print(weight, np.mean(y['err']) * 100, mape)
                # if mape < 0.25:
                #     break
                # # weight += (2*(np.mean(y['err'])<0)-1) / (2**(i+1))
                # weight = np.clip(weight - np.mean(y['err']), -1, 1)
            # weight = min(weight_hist, key=weight_hist.get)
            # print(weight, weight_hist[weight])

            clf_dct = self.param['clf'][2] | {'task':'classification', 'verbose':0, 'log_type': 'all'}
            X = Z.query(f"pred_code=={self.train_code}").copy()
            t = X.query(f"sim==0").groupby([self.crse_code,'__coll_code','__remote'], observed=True)
            y = X.pop(self.crse_code)
            self.weight = 0
            self.hist = dict()
            self.err = 0
            for i in range(20):
                self.weight = np.clip(self.weight - self.err, -1, 1)
                mask = X.reset_index()['index'].isin(t.sample(frac=0.75).reset_index()['index']).values
                dct = clf_dct | {
                    'X_train':X[mask],
                    'y_train':y[mask],
                    'X_val':X[~mask],
                    'y_val':y[~mask],
                    'sample_weight':1+(2*y[mask]-1)*self.weight,
                    'sample_weight_val':1+(2*y[~mask]-1)*self.weight,
                    'log_file_name': self.path.with_stem(f"{self.path.stem}_{i}").with_suffix('.log'),
                }
                mkdir(dct['log_file_name'].parent)
                self.clf = fl.AutoML(**dct)
                self.clf.fit(**dct)
                self.train_score = self.clf.best_result['val_loss'] * 100
                X_all = Z.copy()
                self.Y = X_all.pop(self.crse_code).rename('actual').to_frame().assign(proba=self.clf.predict_proba(X_all)[:,1]).prep(bool=True).copy()
                
                S = self.Y.groupby('pred_code').sum().query(f"pred_code!={self.proj_code}")
                S['proba'] *= S['actual'] > 0
                self.err = S['proba'].sum() / S['actual'].sum() - 1
                self.hist[weight] = self.err
                L = 5
                self.std = np.std(list(self.hist.values())[-L:])
                print(i, self.weight, self.err, self.std)
                S.disp(10)
                if len(self.hist) >= L and self.std < 0.005:
                    break
            

            

            # self.clf = dict()
            # X = Z.query(f"pred_code=={self.train_code}").copy()
            # t = X.query(f"sim==0").groupby([self.crse_code,'__coll_code','__remote'], observed=True)
            # # mask = X.reset_index()['index'].isin(t.sample(frac=0.75).reset_index()['index']).values
            # y = X.pop(self.crse_code)
            # weight = 0
            # for i, z in Z.groupby('sim'):
            # # for z in Z:
            #     a = Z.query(f"pred_code=={self.train_code}").copy()
            #     # trn = a.groupby([self.crse_code,'__coll_code','__remote'], observed=True).sample(frac=0.7)#.reset_index()['index']
            #     # tst = a[~a.isin(trn)]

            #     b = a.groupby([self.crse_code,'__coll_code','__remote'], observed=True).sample(frac=0.7)#.reset_index()['index']
            #     mask = a.isin(b)#.values

            #     clf = fl.AutoML(**dct)
            #     X = Z.copy()
            #     y = X.pop(self.crse_code)
            #     # clf.X = X
            #     # mask = X.eval(f"pred_code=={self.train_code}")
            #     with warnings.catch_warnings(action='ignore'):
            #         clf.fit(
            #             X[mask],
            #             y[mask],
            #             sample_weight=1+(2*y[mask]-1)*weight,
            #             **dct)
            #         clf.Y = y.rename('actual').to_frame().assign(
            #             proba=clf.predict_proba(X)[:,1],
            #             # predicted=clf.predict(clf.X),
            #             ).prep(bool=True).copy()

            #     # mask = a.reset_index()['index'].isin(b).values
            #     # tst = a[~a.isin(b)]


            #     # mask = a.reset_index()['index'].isin(b).values
                
            #     # t = z.query(f"sim==0").groupby([self.crse_code,'__coll_code','__remote'], observed=True)
            #     # mask = z.reset_index()['index'].isin(t.sample(frac=0.7).reset_index()['index']).values


                # mask = z.eval(f"pred_code=={self.train_code}")
                # clf = train(z, mask, weight)
            # self.weight = weight
            # self.train_score = self.clf.best_result['val_loss'] * 100
            # S = self.Y.groupby(['crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim']).apply(lambda y: pd.Series({
            #     'actual': y['actual'].sum(),
            #     # 'predicted': y['predicted'].sum(),
            #     'predicted': y['proba'].sum(),
            #     'train_score': self.train_score,
            #     'test_score': log_loss(y['actual'], y['proba'], labels=[False,True], sample_weight=1+(2*y['actual']-1)*self.weight) * 100,
            #     # 'test_score': (1 - f1_score(y['actual'], y['predicted'], sample_weight=1+(2*y['actual']-1)*clf.weight))*100,
            #     'weight': self.weight,
            # })).prep()
            # S.disp(5)
            # proj_mask = S.eval(f"pred_code==@self.proj_code")
            # proj_col = f'{self.proj_code}_projection'
            # S[proj_col] = S.loc[proj_mask, 'predicted'].squeeze()
            # S.disp(5)
            # S = S[~proj_mask].join(self.mlt.query(f"crse_code==@self.crse_code").squeeze().rename('mlt'))
            # S.disp(5)
            # for k in ['predicted','actual',proj_col]:
            #     S[k] *= S['mlt']
            # S.disp(5)
            # alpha = 1
            # S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
            # S['error'] = S['predicted'] - S['actual']
            # S['error_pct'] = S['error'] / S['actual'] * 100
            # # return S[[proj_col,'overall_score','error_pct','error','predicted','actual','test_score','train_score']]
            # self.summary = S[[proj_col,'overall_score','error_pct','error','predicted','actual','test_score','train_score','weight']]




            # clf.summary = self.summarize(clf)
            # self.clf[i] = clf
            #     # self.clf.append(clf)
            # for k in ['Y', 'summary']:
            #     # self[k] = pd.concat([getattr(clf, k) for clf in self.clf])
            #     self[k] = pd.concat([clf.__dict__.pop(k) for clf in self.clf.values()])
            # self.clf = {i: clf._trained_estimator for i, clf in self.clf.items()}
                
            # grp = [k for k in self.summary.index.names if k!= 'sim']
            # self.rslt = {str(stat): self.summary.groupby(grp).agg(stat) for stat in self.stats}
            # self.rslt[' 50%'].disp(100)
            # del self.X_proc
            # del self.mlt
        return self.get(func, f"Y/{self.styp_code}/{self.crse_code}/{self.train_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl", "X_proc")

    # def get_Y(self):
    #     def func():
    #         cols = uniquify(['_allcrse_cur', self.crse_code+'_cur', self.crse_code])
    #         Y = self.Y.query(f"crse_code.isin({cols})").unstack().droplevel(0,1)
    #         if self.X_proc[0].query(f"pred_code=={self.train_code}").join(Y)[self.crse_code].sum() < 10:
    #             return 'fail'
    #         Z = [
    #             X.join(Y)
    #             .fillna({c:0 for c in cols})
    #             .addlevels({'crse_code':self.crse_code, 'train_code':self.train_code, 'clf_hash':self.param['clf'][0]})
    #             .prep(bool=True, cat=True)
    #             for X in self.X_proc]
    #         clf_dct = self.param['clf'][2].copy() | {'split_type':'stratified', 'task':'classification', 'verbose':0, 'log_file_name': self.path.with_suffix('.log')}
    #         n_calibrations = clf_dct.pop('n_calibrations')
    #         clf_dct['log_file_name'].parent.mkdir(exist_ok=True, parents=True)

    #         def train(Z, mask, weight=0, dct=clf_dct):
    #             clf = fl.AutoML(**dct)
    #             X = Z.copy()
    #             y = X.pop(self.crse_code)
    #             # clf.X = X
    #             # mask = X.eval(f"pred_code=={self.train_code}")
    #             with warnings.catch_warnings(action='ignore'):
    #                 clf.fit(
    #                     X[mask],
    #                     y[mask],
    #                     sample_weight=1+(2*y[mask]-1)*weight,
    #                     **dct)
    #                 clf.Y = y.rename('actual').to_frame().assign(
    #                     proba=clf.predict_proba(X)[:,1],
    #                     # predicted=clf.predict(clf.X),
    #                     ).prep(bool=True).copy()
    #             return clf

    #         print()
    #         weight = 0
    #         weight_hist = {weight: np.inf}
    #         for i in range(n_calibrations):
    #             z = Z[0].query(f"pred_code=={self.train_code}")
    #             t = z.groupby([self.crse_code,'__coll_code','__remote'], observed=True).sample(frac=0.7)


    #             z = Z[i%len(Z)].query(f"pred_code=={self.train_code}")
    #             t = z.groupby([self.crse_code,'__coll_code','__remote'], observed=True).sample(frac=0.7)
    #             mask = z.assign(mask=z.index.isin(t.index)).pop('mask')
    #             clf = train(z, mask, weight, clf_dct|{'time_budget':15})
    #             y = clf.Y[~mask].sum()

    #             # clf = train(Z[i%len(Z)], weight, clf_dct|{'time_budget':15})
    #             # y = clf.Y[~mask].groupby('pred_code').sum().query(f"pred_code!={self.proj_code}")
    #             # y['proba'] *= (y['actual'] > 0)
    #             err = y['proba'].sum() / y['actual'].sum() - 1
    #             print(y, err)
    #             weight_hist[weight] = abs(err)
    #             print(weight, err)
    #             if abs(err) < 0.001:
    #                 break
    #             # weight += (2*(np.mean(y['err'])<0)-1) / (2**(i+1))
    #             weight = np.clip(weight - err, -1, 1)

    #             # y['err'] = np.where(y['actual']<10, pd.NA, y['proba'] / y['actual'] - 1)
    #             # y.disp(5)
    #             # mape = np.mean(np.abs(y['err'])) * 100
    #             # weight_hist[weight] = mape
    #             # print(weight, np.mean(y['err']) * 100, mape)
    #             # if mape < 0.25:
    #             #     break
    #             # # weight += (2*(np.mean(y['err'])<0)-1) / (2**(i+1))
    #             # weight = np.clip(weight - np.mean(y['err']), -1, 1)
    #         weight = min(weight_hist, key=weight_hist.get)
    #         print(weight, weight_hist[weight])

    #         self.clf = []
    #         for z in Z:
    #             mask = z.eval(f"pred_code=={self.train_code}")
    #             clf = train(z, mask, weight)
    #             clf.weight = weight
    #             clf.train_score = clf.best_result['val_loss'] * 100
    #             clf.summary = self.summarize(clf)
    #             self.clf.append(clf)
    #         for k in ['Y', 'summary']:
    #             # self[k] = pd.concat([getattr(clf, k) for clf in self.clf])
    #             self[k] = pd.concat([clf.__dict__.pop(k) for clf in self.clf])
    #         self.clf = [clf._trained_estimator for clf in self.clf]
                
    #         grp = [k for k in self.summary.index.names if k!= 'sim']
    #         self.rslt = {str(stat): self.summary.groupby(grp).agg(stat) for stat in self.stats}
    #         self.rslt[' 50%'].disp(100)
    #         del self.X_proc
    #         del self.mlt
    #     return self.get(func, f"Y/{self.styp_code}/{self.crse_code}/{self.train_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl", "X_proc")


    def summarize(self, clf):
        S = self.Y.groupby(['crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim']).apply(lambda y: pd.Series({
            'actual': y['actual'].sum(),
            # 'predicted': y['predicted'].sum(),
            'predicted': y['proba'].sum(),
            'train_score': self.train_score,
            'test_score': log_loss(y['actual'], y['proba'], labels=[False,True], sample_weight=1+(2*y['actual']-1)*self.weight) * 100,
            'weight': self.weight,
        })).prep()
        proj_mask = S.eval(f"pred_code==@self.proj_code")
        proj_col = f'{self.proj_code}_projection'
        S = (
            S[~proj_mask]
            .join(S[proj_mask]['predicted'].droplevel('pred_code').rename(proj_col))
            .join(self.mlt.squeeze().rename('mlt'))
        )
        for k in ['predicted','actual',proj_col]:
            S[k] *= S['mlt']
        alpha = 1
        S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
        S['error'] = S['predicted'] - S['actual']
        S['error_pct'] = S['error'] / S['actual'] * 100
        return S[[proj_col,'overall_score','error_pct','error','predicted','actual','test_score','train_score']]



        # S = clf.Y.groupby(['crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim']).apply(lambda y: pd.Series({
        #     'actual': y['actual'].sum(),
        #     # 'predicted': y['predicted'].sum(),
        #     'predicted': y['proba'].sum(),
        #     'train_score': clf.train_score,
        #     'test_score': log_loss(y['actual'], y['proba'], labels=[False,True], sample_weight=1+(2*y['actual']-1)*clf.weight) * 100,
        #     # 'test_score': (1 - f1_score(y['actual'], y['predicted'], sample_weight=1+(2*y['actual']-1)*clf.weight))*100,
        #     'weight': clf.weight,
        # })).prep()
        # proj_mask = S.eval(f"pred_code==@self.proj_code")
        # proj_col = f'{self.proj_code}_projection'
        # S[proj_col] = S.loc[proj_mask, 'predicted'].squeeze()
        # S = S[~proj_mask].join(self.mlt.query(f"crse_code==@self.crse_code").squeeze().rename('mlt'))
        # for k in ['predicted','actual',proj_col]:
        #     S[k] *= S['mlt']
        # alpha = 1
        # S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
        # S['error'] = S['predicted'] - S['actual']
        # S['error_pct'] = S['error'] / S['actual'] * 100
        # # return S[[proj_col,'overall_score','error_pct','error','predicted','actual','test_score','train_score']]
        # return S[[proj_col,'overall_score','error_pct','error','predicted','actual','test_score','train_score','weight']]


def get_stack(cycle_day=(Term(term_code=202408).cycle_date-pd.Timestamp.now()).days+1):
    dct = dict()
    append = lambda k, v: dct.setdefault(k,[]).append(v)
    path = AMP(cycle_day=cycle_day).root_path
    for A in (path / 'Y').iterdir():
        for B in A.iterdir():
            for C in B.iterdir():
                for D in C.iterdir():
                    for E in D.iterdir():
                        for F in E.iterdir():
                            if F.suffix == '.pkl':
                                # A = MyBaseClass().load(F)
                                A = read(F)
                                try:
                                    for k in ['Y', 'summary']:
                                        append(k, A[k])
                                    for k, v in A['rslt'].items():
                                        append(k, v)
                                    # print('success')
                                except:
                                    print(F, 'FAILED')
                                    # print('FAILED')
    dct = {k: pd.concat(v).prep() for k, v in dct.items()}
    write(path / 'stack.pkl', dct, overwrite=True)
    print(path / 'stack.pkl')
    return dct

param_grds = {
    'trf': {
        'act_equiv': 'passthrough',
        'act_equiv_missing': 'passthrough',
        'admt_code': 'drop',
        'apdc_day': 'passthrough',
        'appl_day': 'drop',
        'birth_day': 'passthrough',
        'camp_code': 'drop',
        'coll_code': 'passthrough',
        # 'distance': ['passthrough', pwrtrf],
        'distance': [pwrtrf],
        'fafsa_app': 'drop',
        'finaid_accepted': 'drop',
        'gap_score': 'passthrough',
        'gender': 'passthrough',
        'hs_qrtl': 'passthrough',
        'international': 'passthrough',
        'lgcy': 'passthrough',
        'math': 'passthrough',
        'oriented': 'passthrough',
        'pred_code': 'drop',
        **{f'race_{r}': 'passthrough' for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'reading': 'passthrough',
        'remote': 'passthrough',
        'resd': 'passthrough',
        'schlship_app': 'passthrough',
        'ssb': 'passthrough',
        'styp_code': 'drop',
        'waiver': 'passthrough',
        'writing': 'passthrough',
    },
    'imp': {
        'random_state': 42,
        'datasets': 10,
        'iterations': 4,
        # 'datasets': 2,
        # 'iterations': 1,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'seed': 42,
        # 'metric': F_beta(1),
        # 'metric': 'f1',
        'metric': 'log_loss',
        'early_stop': True,
        'time_budget': 30,
        # 'time_budget': 5,
        'estimator_list': [['lgbm','xgboost']],#,'catboost','histgb','extra_tree','xgb_limitdepth','rf','lrl1','lrl2','kneighbor'
        # 'ensemble': [False, True],
        'ensemble': False,
        # 'eval_method': 'cv',
        # 'n_splits': 5,
        # 'n_calibrations':15,
        # 'n_calibrations':12,
    },
}


formatter = lambda x: str(x).replace('\n','').replace(' ','')
hasher = lambda x, d=2: hashlib.shake_128(formatter(x).encode()).hexdigest(d)
param_dcts = dict()
for key, val in param_grds.items():
    lst = cartesian(val, sort=True, key=str)
    if key == 'trf':
        lst = [[(c,t,["__"+c]) for c,t in trf.items() if t not in ['drop', None, '']] for trf in lst]
    param_dcts[key] = [[hasher(k), formatter(k), k] for k in lst]

overwrite = [
    # 'adm',
    # 'flg',
    # 'raw',
    # 'reg',
    # 'X',
    # 'X_proc',
    # 'Y',
]

def run_amp(cycle_day=None):
    for styp_code in ['n']:
        for crse_code in crse_codes:
            for train_code in [202108, 202208, 202308]:
            # for train_code in [202308]:
                for param in cartesian(param_dcts):
                    self = AMP(
                        styp_code=styp_code,
                        crse_code=crse_code,
                        train_code=train_code,
                        param=param,
                        overwrite=overwrite,
                        cycle_day=cycle_day,
                    )
                    self.get_Y()
    # return get_stack(self.cycle_day)
    return self



if __name__ == "__main__":
    print(pd.Timestamp.now())
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        # display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
        # print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))
        print(df.head(max_rows).to_markdown(tablefmt='psql'))
    from IPython.utils.io import Tee
    with contextlib.closing(Tee('/home/scook/institutional_data_analytics/admitted_matriculation_projection/admitted_matriculation_predictor/log.txt', "w", channel="stdout")) as outputstream:
        run_amp(134)