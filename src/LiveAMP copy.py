from term import *
import requests, miceforest as mf
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn import set_config
set_config(transform_output="pandas")

stringify = lambda x: str(x).replace('\n','').replace(' ','')

def feature_importance_df(self, dataset=0, iteration=None, normalize=True):
    targ = [self._get_var_name_from_scalar(int(i)) for i in np.sort(self.imputation_order)]
    feat = [self._get_var_name_from_scalar(int(i)) for i in np.sort(self.predictor_vars)]
    I = pd.DataFrame(self.get_feature_importance(dataset, iteration), index=targ, columns=feat).T
    return I / I.sum() * 100 if normalize else I
mf.ImputationKernel.feature_importance_df = feature_importance_df

def inspect(self, **kwargs):
    self.plot_imputed_distributions(wspace=0.3,hspace=0.3)
    plt.show()
    self.plot_mean_convergence(wspace=0.3, hspace=0.4)
    plt.show()
    I = self.feature_importance_df(**kwargs)
    I.disp(100)
    return I
mf.ImputationKernel.inspect = inspect


@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day: int
    term_codes: typing.List
    pred_code: int
    crse_codes: typing.List
    attr: typing.List
    styp_codes: tuple = ('n','t','t')
    fill: typing.Dict = None
    trf_grid: typing.Dict = None
    imp_grid: typing.Dict = None
    clf_grid: typing.Dict = None
    overwrite: typing.Dict = None
    show: typing.Dict = None

    def __post_init__(self):
        self.root = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}"
        mkdir(self.root)
        self.dependancy = {
            'raw':['reg','adm','flg'],
            'terms':[],
            'raw_df':'terms',
            'reg_df':'terms',
            'X':'raw_df',
            'Y':['X','reg_df'],
            'mlt':['X','reg_df'],
            'transformed':'X',
            'imputed':'transformed',
            'predicted':['Y','imputed'],
            'optimal':'predicted',
            'details':'optimal',
            'summary':['details','mlt'],
            'params':'predicted',
        }
        D = {'trm':False, 'adm':False, 'reg':False, 'flg':False, 'raw':False, 
             'terms':False, 'raw_df':False, 'reg_df':False, 'X':False, 'Y':False, 'mlt':False, 'inputs':False,
             'transformed':False, 'imputed':False, 'predicted':False, 'optimal':False,
             'details':False, 'summary':False, 'params':False, 'outputs':False,
             }
        for x in ['overwrite','show']:
            self[x] = D.copy() if self[x] is None else D.copy() | self[x]
        # self.overwrite['inputs'] |= self.overwrite['X'] | self.overwrite['Y'] | self.overwrite['mlt']
        # self.overwrite['outputs'] |= self.overwrite['details'] | self.overwrite['summary'] | self.overwrite['params']
        # for dep, L in self.dependancy.items():
        #     for ind in listify(L):
        #         self.overwrite[dep] |= self.overwrite[ind]
        for k, v in self.overwrite.items():
            if v:
                delete(self.root / k)
        for k in ['fill','trf_grid','imp_grid']:
            if k not in self:
                self[k] = dict()

        self.crse_codes = uniquify(['_allcrse', *listify(self.crse_codes)])
        self.styp_codes = uniquify(self.styp_codes)
        self.term_codes = [x for x in uniquify(self.term_codes) if x != self.pred_code]

        self.trf_list = cartesian({k: uniquify(v, key=str) for k,v in self.trf_grid.items()})
        self.trf_list = [uniquify({k:v for k,v in t.items() if v not in ['drop',None,'']}) for t in self.trf_list]

        imp_default = {'datasets':5, 'iterations':3, 'tune':True, 'mmc':0, 'mmf':None}
        self.imp_list = cartesian(self.imp_grid)
        self.imp_list = [uniquify(imp_default|v) for v in self.imp_list]
        
        clf_default = {'datasets':5, 'iterations':3, 'tune':True, 'mmc':0, 'mmf':None}
        self.clf_list = cartesian(self.clf_grid)
        self.clf_list = [uniquify(clf_default | v) for v in self.clf_list]

    def get_filename(self, path, suffix='.pkl'):
        return (self.root / join(path.values() if isinstance(path, dict) else path, '/')).with_suffix(suffix)

    def get(self, path, val=None, **kwargs):
        if val is not None:
            nest(path, self.__dict__, val)
            write(self.get_filename(path, **kwargs), val, overwrite=True)
        try:
            val = nest(path, self.__dict__)
        except:
            try:
                val = read(self.get_filename(path))
                nest(path, self.__dict__, val)
            except:
                val = None
        return val

    def run(self, grid, func):
        start_time = time.perf_counter()
        I = cartesian(grid, sort=False)
        for i, path in enumerate(I):
            A = self.get(path)
            if A is None:
                [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[path['nm']])]
                print(join(path.values()), end="; ")
                A = func(path.copy())
                elapsed = (time.perf_counter() - start_time) / 60
                complete = (i+1) / len(I) if len(I) > 0 else 1
                rate = elapsed / (i+1)
                remaining = rate * (len(I) - (i+1))
                self.get(path, A)
                print(f"complete: {(i+1)} / {len(I)} = {complete*100:.2f}%; elapsed = {elapsed:.2f} min; remaining = {remaining:.2f} min @ {rate:.4f} min per iteration")
            # else:
            #     print(path)
        # clear_output()
        return A

    def get_terms(self):
        # grid = {'grp':'inputs', 'nm':'terms', 'term_code':uniquify([*self.term_codes, self.pred_code])}
        grid = {'nm':'terms', 'term_code':uniquify([*self.term_codes, self.pred_code])}
        def func(path):
            opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
            A = TERM(term_code=path['term_code'], **opts).get_raw()
            return A
        self.run(grid, func)

    def where(self, df):
        # return df.query("levl_code == 'ug' and styp_code in @self.styp_codes").copy().rename(columns={'term_code':'pred_code', 'term_desc':'pred_desc'})
        return df.query("levl_code == 'ug' and styp_code in ('n','r','t')").copy().rename(columns={'term_code':'pred_code', 'term_desc':'pred_desc'})

    def get_raw_df(self):
        # grid = {'grp':'inputs', 'nm':'raw_df'}
        grid = {'nm':'raw_df', 'styp_code':'all'}
        def func(path):
            with warnings.catch_warnings(action='ignore'):
                # A = self.where(pd.concat([term.raw for term in self['inputs']['terms'].values()], ignore_index=True).dropna(axis=1, how='all')).prep()
                A = self.where(pd.concat([term.raw for term in self['terms'].values()], ignore_index=True).dropna(axis=1, how='all')).prep()
            return A
        self.run(grid, func)

    def get_reg_df(self):
        # grid = {'grp':'inputs', 'nm':'reg_df'}
        grid = {'nm':'reg_df', 'styp_code':'all'}
        def func(path):
            with warnings.catch_warnings(action='ignore'):
                # A = {k: self.where(pd.concat([term.reg[k].query(f"crse_code in {self.crse_codes}") for term in self['inputs']['terms'].values()])).prep().set_index(['pidm','pred_code','crse_code']) for k in ['cur','end']}
                A = {k: self.where(pd.concat([term.reg[k].query(f"crse_code in {self.crse_codes}") for term in self['terms'].values()])).prep().set_index(['pidm','pred_code','crse_code']) for k in ['cur','end']}
            return A
        self.run(grid, func)

    def get_X(self):
        # grid = {'grp':'inputs', 'nm':'X'}
        grid = {'nm':'X', 'styp_code':'all'}
        def func(path):
            # R = self['inputs']['raw_df']
            R = self['raw_df']['all']
            repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
            R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
            R['remote'] = R['camp_code'] != 's'
            R['resd'] = R['resd_code'] == 'r'
            R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
            R['majr_code'] = R['majr_code'].replace({'0000':pd.NA, 'und':pd.NA, 'eled':'eted', 'agri':'unda'})
            R['coll_code'] = R['coll_code'].replace({'ae':'an', 'eh':'ed', 'hs':'hl', 'st':'sm', '00':pd.NA})
            # R['coll_desc'] = R['coll_desc'].replace({
            #     'ag & environmental_sciences':'ag & natural_resources',
            #     'education & human development':'education',
            #     'health science & human_service':'health sciences',
            #     'science & technology':'science & mathematics'})
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
            X = R.drop(columns=majr).merge(S, on='majr_code', how='left').prep().prep_bool()

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
            for k, v in self.fill.items():
                X[k] = X.impute(k, *listify(v))
            M = X.isnull().rename(columns=lambda x:x+'_missing')
            A = X.join(M).prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x)
            return A
        self.run(grid, func)

    def get_Y(self):
        grid = {'nm':'Y', 'styp_code':'all'}
        def func(path):
            Y = {k: self['X']['all'][[]].join(y)['credit_hr'].unstack().dropna(how='all', axis=1).fillna(0) for k, y in self['reg_df']['all'].items()}
            Y = {k: y.assign(**{k:0 for k in self.crse_codes if k not in y.columns}) for k, y in Y.items()}
            A = Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()
            return A
        self.run(grid, func)

    def get_mlt(self):
        # grid = {'grp':'inputs', 'nm':'mlt'}
        grid = {'nm':'mlt', 'styp_code':'all'}
        def func(path):
            mlt_grp = ['crse_code','levl_code','styp_code','pred_code']
            Y = {k: self['X']['all'][[]].join(y)[['credit_hr']] for k, y in self['reg_df']['all'].items()}
            agg = lambda y: y.query(f"pred_code != {self.pred_code}").groupby(mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())#.rename('mlt')#.to_frame()
            numer = agg(self['reg_df']['all']['end'])
            denom = agg(Y['end'])
            M = (numer / denom).replace(np.inf, pd.NA).reset_index('pred_code').prep()#.squeeze()#.query(f"pred_code != {self.pred_code}")
            # M = M.rename(columns={'credit_hr':'predicted_mlt', 'pred_code':'mlt_code'}).join(M.rename(columns={'credit_hr':'actual_mlt'}))
            M = M.rename(columns={'credit_hr':'actual_mlt'}).join(M.rename(columns={'credit_hr':'predicted_mlt', 'pred_code':'mlt_code'}))

            # M = (numer / denom).replace(np.inf, pd.NA).prep()#.squeeze()#.query(f"pred_code != {self.pred_code}")
            # A = M.rename(columns={'credit_hr':'actual_mlt'}).join(M.rename(columns={'credit_hr':'predicted_mlt'}).rename_axis(index={'pred_code':'mlt_code'}))
            # M = M.rename(columns={'credit_hr':'predicted_mlt'}).rename_axis(index={'pred_code':'mlt_code'}).join(M.rename(columns={'credit_hr':'actual_mlt'}))
            # return M
            # N = M.copy().assign(pred_code=self.pred_code)
            return pd.concat([M, M.assign(pred_code=self.pred_code)]).set_index(['pred_code','mlt_code'], append=True)
            # M = (numer / denom).replace(np.inf, pd.NA).reset_index('pred_code').prep()#.squeeze()#.query(f"pred_code != {self.pred_code}")
            # M.disp(1)
            # # M.rename(columns={'credit_hr':'predicted_mlt', 'pred_code':'mlt_code'}).disp(1)

            # A = M.rename(columns={'credit_hr':'actual_mlt'}).join(M.rename(columns={'credit_hr':'predicted_mlt', 'pred_code':'mlt_code'}))

            # M = (numer / denom).replace(np.inf, pd.NA).prep().squeeze()#.query(f"pred_code != {self.pred_code}")
            # A = M.rename('predicted_mlt').reset_index('pred_code', drop=True).to_frame().join(M.rename('actual_mlt'))
            # A = pd.concat([M.rename('predicted_mlt').reset_index('pred_code', drop=True), M.rename('actual_mlt')], axis=1)
            # agg = lambda y: y.groupby(mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
            # numer = agg(self['reg_df']['all']['end'])
            # denom = agg(Y['end'])
            # M = (numer / denom).replace(np.inf, pd.NA).rename('mlt').reset_index().query(f"pred_code != {self.pred_code}").prep()
            # A = pd.merge(M.rename(columns={'mlt':'predicted_mlt'}).drop(columns='pred_code'), M.rename(columns={'mlt':'actual_mlt'}))
            # print(numer.shape, denom.shape)
            # A = (numer / denom).replace(np.inf, pd.NA).rename('mlt').reset_index('pred_code').query(f"pred_code != {self.pred_code}").prep()
            # L = [A.rename(columns=lambda x: key+x.split('_')[-1]).reset_index() for key in ['actual_','predicted_']]
            # A = L[0].merge(L[1])
            # M['mlt_code'] = M['pred_code']
            # N = M.copy().assign(pred_code=self.pred_code)
            # A = pd.concat([M, N], axis=0).set_index([*mlt_grp,'mlt_code'])
            # A = M.set_index(mlt_grp[:-1])
            # return A
        self.run(grid, func)

    def get_model(self, X, par, inspect=False):
        par = par.copy()
        iterations = par.pop('iterations')
        datasets = par.pop('datasets')
        tune = par.pop('tune')
        mmc = par.pop('mmc')
        mmf = par.pop('mmf')
        if mmc > 0 and mmf is not None:
            par['mean_match_scheme'] = getattr(mf, mmf).copy()
            par['mean_match_scheme'].set_mean_match_candidates(mmc)
        if tune:
            # print('tuning')
            model = mf.ImputationKernel(X, datasets=1, **par)
            model.mice(iterations=1)
            optimal_parameters, losses = model.tune_parameters(dataset=0)#, optimization_steps=3)
        else:
            # print('not tuning')
            optimal_parameters = None
        model = mf.ImputationKernel(X, datasets=datasets, **par)
        model.mice(iterations=iterations, variable_parameters=optimal_parameters)
        if inspect:
            model.inspect()
        return model

    def get_transformed(self):
        grid = {'nm':'transformed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list))}
        # grid = {'nm':'transformed', 'styp_code':self.styp_codes, 'trf_idx': range(len(self.trf_list))}
        def func(path):
            trf_idx = path.pop('trf_idx')
            trf_par = self.trf_list[trf_idx]
            # path = {k:v for k,v in path.items() if k in ['nm','styp_code']}
            trf = ColumnTransformer([(c,t,["__"+c]) for c,t in trf_par.items()], remainder='drop', verbose_feature_names_out=False)
            # trf.trf_idx = trf_idx
            # trf.trf_par = trf_par
            # trf.output = trf.fit_transform(self['inputs']['X'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
            # return trf
            return {
                'idx': trf_idx,
                'par': trf_par,
                'output': trf.fit_transform(self['X']['all'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index(),
                # 'output': trf.fit_transform(self['inputs']['X'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index(),
            }            
        self.run(grid, func)

    def get_imputed(self):
        grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list))}
        # grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list))}
        def func(path):
            imp_idx = path.pop('imp_idx')
            imp_par = self.imp_list[imp_idx]
            # path = {k:v for k,v in path.items() if k in ['nm','styp_code']}
            trf = self.get(path | {'nm':'transformed'})
            imp = self.get_model(trf['output'], imp_par)
            # imp.trf_idx = trf.trf_idx
            # imp.trf_par = trf.trf_par
            # imp.imp_idx = imp_idx
            # imp.imp_par = imp_par
            # imp.output = pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())])
            # return imp
            return {
                'trf': trf,
                'idx': imp_idx,
                'par': imp_par,
                'output': pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())]),
            }
        self.run(grid, func)


    def get_predicted(self):
        grid = {'nm':'predicted', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes, 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list)), 'clf_idx': range(len(self.clf_list))}
        def func(path):
            clf_idx = path.pop('clf_idx')
            clf_par = self.clf_list[clf_idx]
            imp = self.get(path | {'nm':'imputed', 'train_code':'all', 'crse_code':'all'})
            cols = uniquify(['_allcrse_cur', path['crse_code']+'_cur', path['crse_code']], False)
            # Z = imp['output'].join(self['inputs']['Y'][cols]).prep().prep_bool().prep_category().sort_index()
            Z = imp['output'].join(self['Y']['all'][cols]).prep().prep_bool().prep_category().sort_index()
            actual = Z[path['crse_code']].copy().rename('actual').to_frame()
            Z.loc[Z.eval(f"pred_code!=@path['train_code']"), path['crse_code']] = pd.NA
            clf = self.get_model(Z, clf_par)
            # clf.trf_idx = imp.trf_idx
            # clf.trf_par = imp.trf_par
            # clf.imp_idx = imp.imp_idx
            # clf.imp_par = imp.imp_par
            # clf.clf_idx = clf_idx
            # clf.clf_par = clf_par
            # clf.details = pd.concat([actual
            D = pd.concat([actual
                    .assign(predicted=clf.complete_data(k)[path['crse_code']],
                            proba=clf.get_raw_prediction(path['crse_code'], k))
                    .addlevel('crse_code', path['crse_code'])
                    .addlevel('train_code', path['train_code'])
                    .addlevel('sim', k)
                for k in range(clf.dataset_count())]).prep().prep_bool()[['proba','predicted','actual']]
            return self.get_score({
                'imp': imp,
                'idx': clf_idx,
                'par': clf_par,
                'details': D,
            })
        self.run(grid, func)


    def get_score(self, clf):
        Y = clf['details'].query(f"pred_code!=@self.pred_code & pred_code!=train_code")
        # Y.disp(1)
        # func = lambda Y: balanced_accuracy_score(Y['actual'], Y['predicted'])
        # func = lambda Y: accuracy_score(Y['actual'], Y['predicted'])
        with warnings.catch_warnings(action='ignore'):
            clf['score'] = accuracy_score(Y['actual'], Y['predicted'])
            # clf['score'] = Y.apply(func)
            # Z = Y.groupby(['pred_code','sim']).sum()
            # Z['low'] = Z['predicted'] < Z['actual']
            # Z.groupby('pred_code')['low'].mean() - 0.5
            # clf['scores'] = Y.groupby(['pred_code','sim']).apply(func)
            # clf['score'] = clf['scores'].min()
        return clf

    def rescore(self):
        grid = {'nm':'predicted', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes, 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list)), 'clf_idx': range(len(self.clf_list))}
        I = cartesian(grid, sort=False)
        for i, path in enumerate(I):
            print(i, path)
            clf = self.get(path)
            self.get(path, self.get_score(clf))

    def get_optimal(self):
        grid = {'nm':'optimal', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes}
        def func(path):
            D = self.get(path | {'nm':'predicted'})
            E = [A for C in D.values() for B in C.values() for A in B.values()]
            A = max(E, key=lambda e: e['score'])
            return A
        self.run(grid, func)

    def get_details(self):
        # grid = {'grp':'outputs', 'nm':'details'}
        grid = {'nm':'details', 'styp_code':'all'}
        def func(path):
            A = pd.concat([C['details'] for S in self['optimal'].values() for T in S.values() for C in T.values()])
            return A
        self.run(grid, func)

    # def get_optimal(self):
    #     # grid = {'grp':'outputs', 'nm':'details'}
    #     grid = {'nm':'optimal', 'styp_code':'all'}
    #     def func(path):
    #         for clf_list in S in self['predicted'].values() for T in S.values() for C in T.values() for trf in C.values() for imp in trf.values()
    #         # A = pd.concat([C['details'] for S in self['predicted'].values() for T in S.values() for C in T.values() for trf in C.values() for imp in trf.values() for clf. in imp.values()])
    #             A = [max(clf_dict, key=lambda clf: clf['score']) for S in self['predicted'].values() for T in S.values() for C in T.values() for trf_dict in C.values() for imp_dict in trf_dict.values() for clf_dict in imp_dict.values()]
    #         return A
    #     self.run(grid, func)


    def summarize(self, D):
        grp = ['crse_code','levl_code','styp_code','pred_code','train_code','imp','sim']
        agg = lambda X: X.groupby(grp).apply(lambda x: pd.Series({
            'predicted': x['predicted'].sum(),
            'actual': x['actual'].sum(),
            # 'acc_pct': accuracy_score(x['actual'], x['predicted']) * 100,
            'bal_acc_pct': balanced_accuracy_score(x['actual'], x['predicted']) * 100,
            # 'f1_pct': f1_score(x['actual'], x['predicted'], zero_division=np.nan) * 100,
        }), include_groups=False)
        P = [D.reset_index(grp).reset_index(drop=True)]
        # P = [q for p in P for q in [p, p.assign(styp_code='all')]]
        # P = [q for p in P for q in [p, p.query(f"pred_code!={self.pred_code}").assign(train_code='all')]]
        with warnings.catch_warnings(action='ignore'):
            S = pd.concat([agg(p).join(self.mlt['all']) for p in P])
        mask = S.eval(f"train_code=='all'")
        S.disp(2)
        mask.disp(2)
        S.loc[mask, ['predicted','actual']] *= (1.0 / len(self.term_codes))
        for k in ['predicted','actual']:
            S[k] *= S[k+'_mlt']
        S.insert(3, 'error', S['predicted'] - S['actual'])
        S.insert(4, 'error_pct', S['error'] / S['actual'] * 100)
        mask = S.eval(f"actual==0 or pred_code=={self.pred_code}")# or pred_code==train_code")
        S.loc[mask, 'actual':] = pd.NA
        S = S.reset_index()
        S['levl_desc'] = S['levl_code'].map({'ug':'undergraduate', 'g':'graduate'})
        S['styp_desc'] = S['styp_code'].map({'n':'new first time', 't':'transfer', 'r':'returning'})
        for k in ['pred','train','mlt']:
            S[k+'_desc'] = 'Fall ' + S[k+'_code'].astype('string').str[:4]
        return S.set_index(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim'])




        # agg = lambda X: X.groupby(['crse_code','levl_code','styp_code','pred_code','train_code','imp','sim']).apply(lambda x: pd.Series({
        #     'predicted': x['predicted'].sum(),
        #     'actual': x['actual'].sum(),
        #     # 'acc_pct': accuracy_score(x['actual'], x['predicted']) * 100,
        #     'bal_acc_pct': balanced_accuracy_score(x['actual'], x['predicted']) * 100,
        #     # 'f1_pct': f1_score(x['actual'], x['predicted'], zero_division=np.nan) * 100,
        # }), include_groups=False)

        # P = [R for Q in P for R in [Q, Q.assign(styp_code='all', styp_desc='all incoming')]]
        # # P = [R for Q in P for R in [Q, Q.query(f"pred_code!={self.pred_code}").assign(train_code='all', train_desc=f'Fall {min(self.term_codes)}-{max(self.term_codes)}')]]
        # P = [R for Q in P for R in [Q, Q.query(f"pred_code!={self.pred_code}").assign(train_code='all', train_desc=f'{min(self.term_codes)//100}-{max(self.term_codes)//100}')]]


        # # P = D.reset_index(grp).reset_index(drop=True)
        # with warnings.catch_warnings(action='ignore'):
        #     S = agg(D).join(self.mlt['all'])
        # for k in ['predicted','actual']:
        #     S[k] *= S[k+'_mlt']
        # S.insert(3, 'error', S['predicted'] - S['actual'])
        # S.insert(4, 'error_pct', S['error'] / S['actual'] * 100)
        # mask = S.eval(f"actual==0 or pred_code=={self.pred_code}")# or pred_code==train_code")
        # S.loc[mask, 'actual':] = pd.NA
        # S = S.reset_index()
        # S['levl_desc'] = S['levl_code'].map({'ug':'undergraduate', 'g':'graduate'})
        # S['styp_desc'] = S['styp_code'].map({'n':'new first time', 't':'tranafer', 'r':'returning'})
        # for k in ['pred','train','mlt']:
        #     S[k+'_desc'] = 'Fall ' + S[k+'_code'].astype('string').str[:4]
        # S = S.set_index(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim'])

        # P = D.query('pred_code != train_code').join(self['inputs']['mlt']).reset_index()
        # P = D.join(self['inputs']['mlt']).reset_index()
        # P = (D
        #     .join(self['mlt']['all'].rename('actual_mlt'))
        #     .join(self['mlt']['all'].rename('predicted_mlt').reset_index('pred_code', drop=True))
        #     .reset_index()


        # P = D.join(self['mlt']['all']).reset_index()
        # for k in ['pred','train','mlt']:
        #     P[k+'_desc'] = 'Fall ' + P[k+'_code'].astype('string').str[:4]
        # grp = ['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim']
        # P = P[[*grp,'mlt','predicted','actual']]
        # for k in ['predicted','actual']:
        #     P[k+'_mlt'] = P[k] * P['mlt']
        # agg = lambda X: X.groupby(grp).apply(lambda x: pd.Series({
        #         'mlt': x['mlt'].mean(),
        #         'predicted': x['predicted_mlt'].sum(),
        #         'actual': x['actual_mlt'].sum(),
        #         # 'acc_pct': accuracy_score(x['actual'], x['predicted']) * 100,
        #         'bal_acc_pct': balanced_accuracy_score(x['actual'], x['predicted']) * 100,
        #         # 'f1_pct': f1_score(x['actual'], x['predicted'], zero_division=np.nan) * 100,
        #     }), include_groups=False)

        # P = [P]
        # P = [R for Q in P for R in [Q, Q.assign(styp_code='all', styp_desc='all incoming')]]
        # # P = [R for Q in P for R in [Q, Q.query(f"pred_code!={self.pred_code}").assign(train_code='all', train_desc=f'Fall {min(self.term_codes)}-{max(self.term_codes)}')]]
        # P = [R for Q in P for R in [Q, Q.query(f"pred_code!={self.pred_code}").assign(train_code='all', train_desc=f'{min(self.term_codes)//100}-{max(self.term_codes)//100}')]]
        # with warnings.catch_warnings(action='ignore'):
        #     S = pd.concat([agg(Q) for Q in P])
        # mask = S.eval(f"train_code=='all'")
        # S[mask] /= len(self.term_codes)
        # S.insert(3, 'error', S['predicted'] - S['actual'])
        # S.insert(4, 'error_pct', S['error'] / S['actual'] * 100)
        # mask = S.eval(f"actual==0 or pred_code=={self.pred_code}")# or pred_code==train_code")
        # S.loc[mask, 'actual':] = pd.NA
        # return A

    def get_summary(self):
        # grid = {'grp':'outputs', 'nm':'summary'}
        grid = {'nm':'summary', 'styp_code':'all'}
        def func(path):
            D = self.get(path | {'nm':'details'})
            A = self.summarize(D)
            write(self.get_filename(path, suffix='.csv'), A.reset_index().prep_string(cap='upper'))
            return A
        self.run(grid, func)

    def get_params(self):
        # grid = {'grp':'outputs', 'nm':'params'}
        grid = {'nm':'params', 'styp_code':'all'}
        def func(path):
            A = pd.DataFrame([{
                    'pred_code':pred_code, 'crse_code':crse_code, 'styp_code':styp_code, 'train_code':train_code,
                    'trf_idx': clf['imp']['trf']['idx'],
                    'imp_idx': clf['imp']['idx'],
                    'clf_idx': clf['idx'],
                    **{f'trf_{key}': stringify(val) for key, val in clf['imp']['trf']['par'].items()},
                    **{f'imp_{key}': stringify(val) for key, val in clf['imp']['par'].items()},
                    **{f'clf_{key}': stringify(val) for key, val in clf['par'].items()},
                    'score': score,
                } for styp_code, S in self.predicted.items() for train_code, T in S.items() for crse_code, C in T.items() for trf_idx, trf in C.items() for imp_idx, imp in trf.items() for clf_idx, clf in imp.items() for pred_code, score in clf['scores'].items()])
            write(self.get_filename(path, suffix='.csv'), A)
            return A
        self.run(grid, func)

    def push(self):
        target_url = 'https://prod-121.westus.logic.azure.com:443/workflows/784fef9d36024a6abf605d1376865784/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=1Yrr4tE1SwYZ88SU9_ixG-WEdN1GFicqJwH_KiCZ70M'
        path = {'nm':'summary', 'styp_code':'all'}
        with open(self.get_filename(path, suffix='.csv'), 'rb') as target_file:
            response = requests.post(target_url, files = {"amp_summary.csv": target_file})
        print('file pushed')


code_desc = lambda x: [x+'_code', x+'_desc']
bintrf = lambda n_bins: KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())
passthru = ['passthrough']

passdrop = ['passthrough', 'drop']
# passdrop = passthru

passpwr = ['passthrough', pwrtrf]
# passpwr = passthru

kwargs = {
    'term_codes': np.arange(2023,2024)*100+8,
    'pred_code': 202408,
    'show': {
        # 'reg':True,
        # 'adm':True,
    },
    'fill': {
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
        'oriented': 'n',
    },
    'attr': [
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
    ],
    'cycle_day': (TERM(term_code=202408).cycle_date-pd.Timestamp.now()).days+1,
    'cycle_day': 165,
    'crse_codes': [
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
        ],
    'trf_grid': {
        'act_equiv': passthru,
        # 'act_equiv_missing': passdrop,
        'act_equiv_missing': passthru,
        # 'admt_code': passdrop,
        'apdc_day': passthru,
        # 'appl_day': passthru,
        # 'birth_day': passpwr,
        'birth_day': passthru,
        # 'camp_code': passdrop,
        'coll_code': passthru,
        # 'coll_code': passdrop,
        # 'coll_desc': passdrop,
        'distance': passpwr,
        # 'fafsa_app': passthru,
        # 'finaid_accepted': passthru,
        'gap_score': passthru,
        'gender': passthru,
        'hs_qrtl': passthru,
        'international': passthru,
        'lgcy': passthru,
        'math': passthru,
        'oriented': passthru,
        'pred_code': passthru,
        **{f'race_{r}': passthru for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'reading': passthru,
        'remote': passthru,
        'resd': passthru,
        'schlship_app': passthru,
        'ssb': passthru,
        # 'styp_code': passthru,
        'waiver': passthru,
        'writing': passthru,
        },
    'imp_grid': {
        # 'datasets': 2, 'iterations': 1, 'tune': False,
    },
    'clf_grid': {
        # 'datasets': 2, 'iterations': 1, 'tune': False,
    },

    'overwrite': {
        # 'trm':True,
        # 'reg':True,
        # 'adm':True,
        # 'flg':True,
        # 'raw':True,
        # 'terms': True,
        'raw_df': True,
        'reg_df': True,
        'X': True,
        'Y': True,
        'mlt': True,
        ## 'inputs': True,
        # 'transformed': True,
        # 'imputed': True,
        # 'predicted': True,
        # 'optimal': True,
        # 'details': True,
        'summary': True,
        # 'params': True,
        ## 'outputs': True,
    },
    'styp_codes': ['n'],
}

if __name__ == "__main__":
    print(pd.Timestamp.now())

    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
        print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    self = AMP(**kwargs)
    with contextlib.closing(Tee(self.root / 'log.txt', "w", channel="stdout")) as outputstream:
        print(self.styp_codes)
        print(self.crse_codes)
        self.get_predicted()
        # self.get_summary()
        # self.get_params()
        # self.push()