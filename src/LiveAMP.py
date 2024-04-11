from term import *
import requests, miceforest as mf#, flaml as fl
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, auc
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
    n_splits: int = 3
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
            'anonymized':'transformed',
            'imputed':'transformed',
            'targets':['imputed','Y'],
            'predicted':'targets',
            'optimal':'predicted',
            'details':'optimal',
            'summary':['details','mlt'],
            'params':'predicted',
        }
        D = {'trm':False, 'adm':False, 'reg':False, 'flg':False, 'raw':False, 
             'terms':False, 'raw_df':False, 'reg_df':False, 'X':False, 'Y':False, 'mlt':False, 'inputs':False,
             'transformed':False, 'imputed':False, 'targets':False, 'predicted':False, 'optimal':False,
             'details':False, 'summary':False, 'params':False, 'outputs':False,
             }
        for x in ['overwrite','show']:
            self[x] = D.copy() if self[x] is None else D.copy() | self[x]
        for k, v in self.overwrite.items():
            if v:
                delete(self.root / k)
        for k in ['fill','trf_grid','imp_grid']:
            if k not in self:
                self[k] = dict()

        self.crse_codes = uniquify(['_allcrse', *listify(self.crse_codes)])
        self.styp_codes = uniquify(self.styp_codes)
        self.mlt_grp = ['crse_code','levl_code','styp_code','pred_code']
        self.summary_grp = self.mlt_grp + ['train_code','mlt_code','imp']
        self.term_codes = [x for x in uniquify(self.term_codes) if x != self.pred_code]

        self.trf_list = cartesian({k: uniquify(v, key=str) for k,v in self.trf_grid.items()})
        self.trf_list = [uniquify({k:v for k,v in t.items() if v not in ['drop',None,'']}) for t in self.trf_list]

        # imp_default = {'datasets':10, 'iterations':3, 'tune':True, 'mmc':0, 'mmf':None}
        imp_default = {'datasets':10, 'iterations':5, 'tune':False, 'mmc':0, 'mmf':None}
        self.imp_list = cartesian(self.imp_grid)
        self.imp_list = [uniquify(imp_default|v) for v in self.imp_list]
        
        clf_default = {'datasets':1, 'iterations':5, 'tune':False, 'mmc':0, 'mmf':None}
        # clf_default = {'time_budget':5}
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

    def run(self, grid, func, **kwargs):
        start_time = time.perf_counter()
        I = cartesian(grid, sort=False)
        for i, path in enumerate(I):
            A = self.get(path)
            if A is None:
                [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[path['nm']])]
                print(join(path.values()), end="; ")
                A = func(path.copy())
                self.get(path, A, **kwargs)
                elapsed = (time.perf_counter() - start_time) / 60
                complete = (i+1) / len(I) if len(I) > 0 else 1
                rate = elapsed / (i+1)
                remaining = rate * (len(I) - (i+1))
                print(f"complete: {(i+1)} / {len(I)} = {complete*100:.2f}%; elapsed = {elapsed:.2f} min; remaining = {remaining:.2f} min @ {rate:.4f} min per iteration")
        return A

    def get_terms(self):
        grid = {'nm':'terms', 'term_code':range(202108, self.pred_code+1, 100)}
        def func(path):
            opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
            A = TERM(term_code=path['term_code'], **opts).get_raw()
            return A
        self.run(grid, func)

    def where(self, df):
        return df.query("levl_code == 'ug' and styp_code in ('n','r','t')").copy().rename(columns={'term_code':'pred_code', 'term_desc':'pred_desc'})

    def get_raw_df(self):
        grid = {'nm':'raw_df', 'styp_code':'all'}
        def func(path):
            with warnings.catch_warnings(action='ignore'):
                A = self.where(pd.concat([term.raw for term in self['terms'].values()], ignore_index=True).dropna(axis=1, how='all')).prep()
            return A
        self.run(grid, func)

    def get_reg_df(self):
        grid = {'nm':'reg_df', 'styp_code':'all'}
        def func(path):
            with warnings.catch_warnings(action='ignore'):
                A = {k: self.where(pd.concat([term.reg[k].query(f"crse_code in {self.crse_codes}") for term in self['terms'].values()])).prep().set_index(['pidm','pred_code','crse_code']) for k in ['cur','end']}
            return A
        self.run(grid, func)

    def get_X(self):
        grid = {'nm':'X', 'styp_code':'all'}
        def func(path):
            R = self['raw_df']['all']
            repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
            R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
            R['remote'] = R['camp_code'] != 's'
            R['resd'] = R['resd_code'] == 'r'
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
            # A = X.join(M).reset_index().rename_axis('idx').prep().prep_bool().set_index(self.attr, drop=False, append=True).rename(columns=lambda x:'__'+x)
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
        grid = {'nm':'mlt', 'styp_code':'all'}
        def func(path):
            Y = {k: self['X']['all'][[]].join(y)[['credit_hr']] for k, y in self['reg_df']['all'].items()}
            agg = lambda y: y.query(f"pred_code != {self.pred_code}").groupby(self.mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
            numer = agg(self['reg_df']['all']['end'])
            denom = agg(Y['end'])
            A = (numer / denom).replace(np.inf, pd.NA).reset_index('pred_code').prep()
            B = A.rename(columns={'pred_code':'train_code', 'credit_hr':'mlt_actual'})
            C = A.rename(columns={'pred_code':'mlt_code'  , 'credit_hr':'mlt_predicted'})
            D = B.join(C)
            M = pd.concat([D.assign(pred_code=t) for t in [*A['pred_code'].unique(), self.pred_code]]).set_index(['pred_code','train_code','mlt_code'], append=True).filter(like='mlt_')


            # Y = {k: self['X']['all'][[]].join(y)[['credit_hr']] for k, y in self['reg_df']['all'].items()}
            # agg = lambda y: y.query(f"pred_code != {self.pred_code}").groupby(self.mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())#.rename('mlt')#.to_frame()
            # numer = agg(self['reg_df']['all']['end'])
            # denom = agg(Y['end'])
            # M = (numer / denom).replace(np.inf, pd.NA).reset_index('pred_code').prep()
            # M = M.rename(columns={'credit_hr':'actual_mlt'}).join(M.rename(columns={'credit_hr':'predicted_mlt', 'pred_code':'mlt_code'})).assign(train_code=lambda x:x['pred_code'])
            # A = pd.concat([M, M.assign(pred_code=self.pred_code)]).reset_index().set_index([*self.mlt_grp,'train_code','mlt_code'])
            return M
        self.run(grid, func)

    def get_transformed(self):
        grid = {'nm':'transformed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list))}
        def func(path):
            trf_idx = path.pop('trf_idx')
            trf_par = self.trf_list[trf_idx]
            trf = ColumnTransformer([(c,t,["__"+c]) for c,t in trf_par.items()], remainder='drop', verbose_feature_names_out=False)
            return {
                'trf_idx': trf_idx,
                'trf_par': trf_par,
                'output': trf.fit_transform(self['X']['all'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category(),
            }            
        self.run(grid, func)


    def get_anonymized(self):
        A = [[len(par), i] for i, par in enumerate(self.trf_list) if set(par.values()) == {'passthrough'}]
        trf_idx = sorted(A)[-1][-1]
        grid = {'nm':'anonymized', 'styp_code':'n', 'train_code':'all', 'crse_code':self.crse_codes}
        def func(path):
            targ = path['crse_code']
            cols = ['_allcrse_cur', targ+'_cur', targ]
            grp = ['styp_code','pred_code','coll_code','coll_desc','camp_code','camp_desc']
            X = (
                self.get(path | {'nm':'transformed', 'crse_code':'all', 'trf_idx':trf_idx})['output']
                .rename(columns=lambda x:x.strip('__'))
                .drop(columns=[*grp, 'remote'], errors='ignore')
                .join(self['Y']['all'].filter(cols))
                .reset_index(grp)
                .reset_index(drop=True)
                .sample(frac=1).prep().prep_bool()#.prep_category()
            )
            return X
        return self.run(grid, func, suffix='.parquet')
        # write(self.root / 'amp.parquet', X)
        # return X


    def get_model(self, X, datasets=1, iterations=3, mmc=0, mmf=None, tune=False, inspect=False):
        mean_match_scheme = None
        if mmc > 0 and mmf is not None:
            mean_match_scheme = getattr(mf, mmf).copy()
            mean_match_scheme = mean_match_scheme.set_mean_match_candidates(mmc)
        if tune:
            # print('tuning')
            model = mf.ImputationKernel(X, datasets=1, mean_match_scheme=mean_match_scheme)
            model.mice(iterations=iterations)
            optimal_parameters, losses = model.tune_parameters(dataset=0, optimization_steps=10)
        else:
            # print('not tuning')
            optimal_parameters = None
        model = mf.ImputationKernel(X, datasets=datasets, mean_match_scheme=mean_match_scheme)
        model.mice(iterations=iterations, variable_parameters=optimal_parameters)
        if inspect:
            model.inspect()
        return model


    def get_imputed(self):
        grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list))}
        def func(path):
            imp_idx = path.pop('imp_idx')
            imp_par = self.imp_list[imp_idx]
            trf = self.get(path | {'nm':'transformed'})
            imp = self.get_model(trf['output'], **imp_par)
            return {
                'trf_idx': trf['trf_idx'],
                'trf_par': trf['trf_par'],
                'imp_idx': imp_idx,
                'imp_par': imp_par,
                'output': pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())]),
            }
        self.run(grid, func)


    def get_targets(self):
        grid = {'nm':'targets', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list))}
        def func(path):
            imp = self.get(path | {'nm':'imputed'})
            imp['output'] = imp['output'].filter(like='__').join(self['Y']['all']).sample(frac=1).prep().prep_bool().prep_category()
            return imp
        self.run(grid, func)


    def summarize(self, Y):
        repl = {'mlt_'+k:k for k in ['predicted','actual']} | {'mlt_code':'train_code'}
        for k, v in repl.items():
            if k not in Y:
                Y[k] = Y[v]
        S = Y.groupby(self.summary_grp, dropna=False).apply(lambda y: pd.Series({
            'predicted': y['mlt_predicted'].sum(),
            'actual': y['mlt_actual'].sum(),
            'f1': (1-f1_score(y['actual'], y['predicted'], zero_division=np.nan))*100,
            # 'acc_pct': (1-accuracy_score(y['actual'], y['predicted'])*100),
            # 'bal_acc_pct': )1-balanced_accuracy_score(y['actual'], y['predicted'])*100),
        }), include_groups=False)
        S.insert(2, 'error', S['predicted'] - S['actual'])
        S.insert(3, 'error_pct', S['error'] / S['actual'] * 100)
        return S


    def train(self, df, test_mask, targ, **kwargs):
        Z = df.copy()
        Z_train = Z.copy()
        actual = Z[targ].copy().rename('actual').to_frame()
        Z.loc[:,targ] = pd.NA
        try:
            Z_train.loc[test_mask, targ] = pd.NA
        except:
            Z_train.iloc[test_mask, Z_train.columns.get_loc(targ)] = pd.NA
        with warnings.catch_warnings(action='ignore'):
            model = self.get_model(Z_train, **kwargs)
        predicted = model.impute_new_data(Z)
        Y = (pd.concat([actual
                .assign(
                    proba=predicted.get_raw_prediction(targ, k),
                    predicted=predicted.complete_data(k)[targ],
                    )
                .addlevel('crse_code', targ)
                .addlevel('train_code', Z_train.query(f"{targ}.notnull()").index.get_level_values("pred_code")[0])
                .addlevel('sim', k)
            for k in range(model.dataset_count())])[['predicted','actual']].prep().prep_bool())
        return {'model':model,
                'Y':Y,
                'f1': (1-f1_score(Y['actual'], Y['predicted'], zero_division=np.nan)) *100 ,
                # 'acc': 1-accuracy_score_score(Y['actual'], Y['predicted']),
                # 'bal_acc': 1-balanced_accuracy_score(Y['actual'], Y['predicted']),
                }


    def get_predicted(self):
        grid = {'nm':'predicted', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes, 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list)), 'clf_idx': range(len(self.clf_list))}
        def func(path):
            clf_idx = path.pop('clf_idx')
            clf_par = self.clf_list[clf_idx]
            targ = path['crse_code']
            df = self.get(path | {'nm':'targets', 'train_code':'all', 'crse_code':'all'})
            Z = df['output']
            cols = [*Z.filter(like='__').columns, '_allcrse_cur', targ+'_cur', targ]
            Z = Z.filter(cols).copy()
            clf = {
                'trf_idx': feat['trf_idx'],
                'trf_par': feat['trf_par'],
                'imp_idx': feat['imp_idx'],
                'imp_par': feat['imp_par'],
                'clf_idx': clf_idx,
                'clf_par': clf_par,
                'cv_score': 100,
                'Z': Z,
            }
            Z_model = Z.query(f"pred_code==@path['train_code'] & imp==0")#.copy()
            if Z_model[targ].sum() >= 20:
                splits = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.25).split(Z_model, Z_model[targ])
                clf['cv'] = [self.train(Z, tst, targ, **clf_par) for trn, tst in splits]
                clf['cv_score'] = np.nanmean([c['f1'] for c in clf['cv']])
                print(clf['cv_score'].round(2))
            return clf
        self.run(grid, func)


    def get_optimal(self):
        grid = {'nm':'optimal', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes}
        def func(path):
            C = self.get(path | {'nm':'predicted'})
            E = [clf for trf in C.values() for imp in trf.values() for clf in imp.values() if clf['cv_score'] < 100]
            try:
                clf = min(E, key=lambda clf: clf['cv_score'])
                Z = clf['Z']
                Z.vc('pred_code').disp(100)
                opt = self.train(Z, Z.eval(f"pred_code!=@path['train_code']"), path['crse_code'], **clf['clf_par'])
                opt['cv_score'] = clf['cv_score']
                opt['clf'] = clf
                return opt
            except ValueError:
                return dict()
        self.run(grid, func)


    def get_details(self):
        grid = {'nm':'details', 'styp_code':'all'}
        def func(path):
            A = pd.concat([C['Y'] for S in self['optimal'].values() for T in S.values() for C in T.values() if 'Y' in C and C['cv_score'] < 100])
            return A
        self.run(grid, func)


    def get_summary(self):
        grid = {'nm':'summary', 'styp_code':'all'}
        def func(path):
            Y = self.get(path | {'nm':'details'})
            Y = Y.reset_index([k for k in Y.index.names if k not in self.summary_grp], drop=True).join(self.mlt['all']).reset_index().copy()
            for k in ['predicted','actual']:
                Y['mlt_'+k] = Y['mlt_'+k] * Y[k]
            P = [Y]
            P = [q for p in P for q in [p, p.assign(styp_code ='all')]]
            P = [q for p in P for q in [p, p.assign(train_code='all')]]
            with warnings.catch_warnings(action='ignore'):
                S = pd.concat([self.summarize(p) for p in P])
            S.loc[S.eval("train_code=='all'"), ['predicted','actual','error']] /= Y['train_code'].nunique()
            S.loc[S.eval(f"actual==0 or pred_code=={self.pred_code}"), 'actual':] = pd.NA
            S = S.reset_index()
            S['levl_desc'] = S['levl_code'].map({'ug':'undergraduate', 'g':'graduate'})
            S['styp_desc'] = S['styp_code'].map({'n':'new first time', 't':'transfer', 'r':'returning', 'all':'all incoming'})
            for k in ['pred','train','mlt']:
                S[k+'_desc'] = 'Fall ' + S[k+'_code'].astype('string').str[:4]
            S.vc('pred_code').disp(100)
            S = S.set_index(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp'])
            return S#.reset_index().prep_string(cap='upper')
        self.run(grid, func)
        self.run(grid, func, suffix='.csv')


    # def get_params(self):
    #     grid = {'nm':'params', 'styp_code':'all'}
    #     def func(path):
    #         A = pd.DataFrame([{
    #                 'pred_code':pred_code, 'crse_code':crse_code, 'styp_code':styp_code, 'train_code':train_code,
    #                 'trf_idx': clf['imp']['trf']['idx'],
    #                 'imp_idx': clf['imp']['idx'],
    #                 'clf_idx': clf['idx'],
    #                 **{f'trf_{key}': stringify(val) for key, val in clf['imp']['trf']['par'].items()},
    #                 **{f'imp_{key}': stringify(val) for key, val in clf['imp']['par'].items()},
    #                 **{f'clf_{key}': stringify(val) for key, val in clf['par'].items()},
    #                 'score': score,
    #             } for styp_code, S in self.predicted.items() for train_code, T in S.items() for crse_code, C in T.items() for trf_idx, trf in C.items() for imp_idx, imp in trf.items() for clf_idx, clf in imp.items() for pred_code, score in clf['scores'].items()])
    #         write(self.get_filename(path, suffix='.csv'), A)
    #         return A
    #     self.run(grid, func)


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
passpwr = ['passthrough', pwrtrf]

passdrop = passthru
passpwr = passthru

kwargs = {
    # 'term_codes': np.arange(202308, 202408, 100),
    'term_codes': np.arange(202308, 202408, 100),
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
    # 'cycle_day': 161,
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
        'coll_code': passdrop,
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
        # 'pred_code': passthru,
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
        # 'time_budget': 5,
    },

    'overwrite': {
        # # 'trm':True,
        # # 'reg':True,
        # # 'adm':True,
        # # 'flg':True,
        # # 'raw':True,
        # # 'terms': True,
        # # 'raw_df': True,
        # # 'X': True,
        'reg_df': True,
        'Y': True,
        'mlt': True,
        # 'targets': True,
        # # 'transformed': True,
        # # 'imputed': True,
        # 'predicted': True,
        # 'optimal': True,
        'details': True,
        'summary': True,
        # # 'params': True,
    },
    'styp_codes': ['n','t','r'],
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
        self.get_terms()
        self.get_raw_df()
        self.get_reg_df()
        self.get_X()
        self.get_Y()
        self.get_mlt()
        self.get_transformed()
        self.get_imputed()
        self.get_targets()
        self.get_predicted()
        self.get_optimal()
        self.get_details()
        self.get_summary()
        # self.get_params()
        # self.push()