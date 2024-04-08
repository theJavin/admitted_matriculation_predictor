from term import *
import requests, miceforest as mf
# from flaml import AutoML
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import f1_score
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, auc
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from skopt import BayesSearchCV
# from skopt.callbacks import DeadlineStopper
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

        imp_default = {'datasets':10, 'iterations':5, 'tune':True, 'mmc':0, 'mmf':None}
        self.imp_list = cartesian(self.imp_grid)
        self.imp_list = [uniquify(imp_default|v) for v in self.imp_list]
        
        clf_default = {'datasets':1, 'iterations':5, 'tune':False, 'mmc':0, 'mmf':None}
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
            mlt_grp = ['crse_code','levl_code','styp_code','pred_code']
            Y = {k: self['X']['all'][[]].join(y)[['credit_hr']] for k, y in self['reg_df']['all'].items()}
            agg = lambda y: y.query(f"pred_code != {self.pred_code}").groupby(mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())#.rename('mlt')#.to_frame()
            numer = agg(self['reg_df']['all']['end'])
            denom = agg(Y['end'])
            M = (numer / denom).replace(np.inf, pd.NA).reset_index('pred_code').prep()#.squeeze()#.query(f"pred_code != {self.pred_code}")
            M = M.rename(columns={'credit_hr':'actual_mlt'}).join(M.rename(columns={'credit_hr':'predicted_mlt', 'pred_code':'mlt_code'}))
            A = pd.concat([M, M.assign(pred_code=self.pred_code)]).set_index(['pred_code','mlt_code'], append=True)
            return A
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
            # model.optimal_parameters = optimal_parameters
            # model.optimal_parameter_losses = losses
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
        def func(path):
            trf_idx = path.pop('trf_idx')
            trf_par = self.trf_list[trf_idx]
            trf = ColumnTransformer([(c,t,["__"+c]) for c,t in trf_par.items()], remainder='drop', verbose_feature_names_out=False)
            # trf.idx = idx
            # trf.par = par
            # trf.output = trf.fit_transform(self['X']['all'].query(f"styp_code == @path['styp_code']")).sample(frac=1).prep().prep_bool().prep_category()
            return {
                'trf_idx': trf_idx,
                'trf_par': trf_par,
                'output': trf.fit_transform(self['X']['all'].query(f"styp_code == @path['styp_code']")).sample(frac=1).prep().prep_bool().prep_category(),
                # 'trf': trf,
            }            
        self.run(grid, func)

    def get_imputed(self):
        grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list))}
        def func(path):
            imp_idx = path.pop('imp_idx')
            imp_par = self.imp_list[imp_idx]
            trf = self.get(path | {'nm':'transformed'})
            imp = self.get_model(trf['output'], imp_par)
            # imp.trf = trf
            # imp.idx = idx
            # imp.par = par
            # imp.output = pd.concat([imp.complete_data(k).addlevel('sim', k) for k in range(imp.dataset_count())])
            # Z = pd.concat([imp.complete_data(k).addlevel('sim', k) for k in range(imp.dataset_count())]).join(self['Y']['all']).sample(frac=1)
            return {
                'trf_idx': trf['trf_idx'],
                'trf_par': trf['trf_par'],
                'imp_idx': imp_idx,
                'imp_par': imp_par,
                'output': pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())]).join(self['Y']['all']).sample(frac=1),
                # 'output': pd.concat([imp.complete_data(k).addlevel('sim', k) for k in range(imp.dataset_count())]),
                # 'imp': imp,
            }
        self.run(grid, func)

    def get_predicted(self):
        grid = {'nm':'predicted', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes, 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list)), 'clf_idx': range(len(self.clf_list))}
        def func(path):
            clf_idx = path.pop('clf_idx')
            clf_par = self.clf_list[clf_idx]
            imp = self.get(path | {'nm':'imputed', 'train_code':'all', 'crse_code':'all'})
            clf = {
                'trf_idx': imp['trf_idx'],
                'trf_par': imp['trf_par'],
                'imp_idx': imp['imp_idx'],
                'imp_par': imp['imp_par'],
                'clf_idx': clf_idx,
                'clf_par': clf_par,
                'score': 1.0,
            }
            
            targ = path['crse_code']
            Z = imp['output']
            cols = [*Z.filter(like='__').columns, '_allcrse_cur', targ+'_cur', targ]
            # Z = Z.filter(cols).copy()
            # Y = Z[targ].rename('actual')
            # Z_train = Z.query(f"pred_code==@path['train_code'] & sim==0")
            # x = X.query(f"pred_code==@path['train_code'] & sim==0").copy()
            # Y = X.pop(targ).rename('actual')
            # y = x.pop(targ).rename('actual')
            Z = Z.filter(cols).copy()
            # Z_model = Z.copy()
            actual = Z[targ].copy().rename('actual').to_frame()

            Z_model = Z.query(f"pred_code==@path['train_code'] & imp==0").copy()

            # Z_model.loc[Z_model.eval(f"pred_code!=@path['train_code']"), targ] = pd.NA
            # Z_model[targ] = Z_model[targ].astype('boolean')
            # if y.sum() > 3:
            if Z_model[targ].sum() > 3:
                splits = list(StratifiedKFold(n_splits=3).split(Z_model, Z_model[targ]))
                clf['cv'] = []
                for split in splits:
                    Z_train = Z_model.copy()
                    Z_train.iloc[split[1], Z_train.columns.get_loc(targ)] = pd.NA
                    with warnings.catch_warnings(action='ignore'):
                        model = self.get_model(Z_train, clf_par)
                    Z_train.loc[:,targ] = pd.NA
                    predicted = model.impute_new_data(Z_train)
                    y_test = Z_model[targ].rename('actual').to_frame().assign(predicted=predicted.complete_data(0)[targ]).iloc[split[1]]
                    # predicted.complete_data(0)[targ].rename('predicted').to_frame().assign(actual=Z_model[targ]).iloc[split[1]]
                    # y_test = predicted.complete_data(0)[targ].rename('predicted').to_frame().assign(actual=Z_model[targ]).iloc[split[1]]
                    # y_test = model.complete_data(0)[targ].rename('predicted').to_frame().assign(actual=Z_model[targ]).iloc[split[1]]
                    clf['cv'].append(1-f1_score(y_test['actual'], y_test['predicted'], zero_division=np.nan))
                clf['score'] = np.nanmean(clf['cv'])
                print(clf['cv'], clf['score'])
                #     # print(y.shape)
                
                
                #     print(clf['cv'])
                #     assert 1==2

                #     print(split)
                #     Z_train, actual = Z_model.iloc[split[0]], Z_model.iloc[split[1]]

                #     Z_train.disp(2)
                #     print(Z_train.shape)
                #     actual.disp(2)
                #     print(actual.shape)
                #     # print(clf_par)
                #     model = self.get_model(Z_train, clf_par)
                #     model.complete_data(0).disp(8)
                #     model.impute_new_data(Z_train)
                #     predicted = model.complete_data(0)[targ]
                #     Z_test = actual.copy()
                #     Z_test[targ] = pd.NA
                #     Z_test[targ] = Z_test[targ].astype('boolean')
                #     model.impute_new_data(Z_test)
                #     predicted = model.complete_data(0)[targ]
                #     clf['cv'].append(f1_score(actual, predicted, zero_division=np.nan))
                # clf['score'] = np.nanmean(clf['cv'])
                # assert 1==2


                
                

                # model = mf.ImputationKernel(Z_train, datasets=1)
                # model.mice(iterations=5)
                # optimal_parameters, losses = model.tune_parameters(dataset=0)
                # model.mice(iterations=5, variable_parameters=optimal_parameters)
                # y = pd.concat([y.assign(predicted=model.complete_data(k)[targ]).addlevel('crse_code', targ).addlevel('train_code', path['train_code']).addlevel('sim', k)
                #                for k in range(model.dataset_count())]).prep().prep_bool()[['predicted','actual']]
                # Z[targ] = pd.NA
                # Z[targ] = Z[targ].astype('boolean')
                Z.loc[Z.eval(f"pred_code!=@path['train_code']"), targ] = pd.NA
                with warnings.catch_warnings(action='ignore'):
                    model = self.get_model(Z, clf_par)
                Z.loc[:, targ] = pd.NA
                predicted = model.impute_new_data(Z)
                # y = pd.concat([predicted.complete_data(k)[targ].rename('predicted').to_frame().assign(actual=actual).addlevel('train_code', path['train_code']).addlevel('sim', k)
                #     for k in range(model.dataset_count())]).prep().prep_bool()[['predicted','actual']]

                y = pd.concat([actual.assign(predicted=predicted.complete_data(k)[targ]).addlevel('crse_code', targ).addlevel('train_code', path['train_code']).addlevel('sim', k)
                    for k in range(model.dataset_count())]).prep().prep_bool()[['predicted','actual']]


                # y = pd.concat([y.assign(predicted=model.complete_data(k)[targ]).addlevel('crse_code', targ).addlevel('train_code', path['train_code']).addlevel('sim', k)
                #                for k in range(model.dataset_count())]).prep().prep_bool()[['predicted','actual']]

                # model = self.get_model(Z_model, clf_par)
                # predicted = model.impute_new_data(Z)
                # y = pd.concat([y.assign(predicted=predicted.complete_data(k)[targ]).addlevel('crse_code', targ).addlevel('train_code', path['train_code']).addlevel('sim', k)
                #                for k in range(model.dataset_count())]).prep().prep_bool()[['predicted','actual']]

                # T = y.query(f"pred_code!=@self.pred_code").groupby(['pred_code','sim']).sum()
                # y.disp(2)
                grp = ['crse_code','levl_code','styp_code','pred_code','train_code','imp','sim']
                T = y.groupby(grp).sum()
                T['error'] = T['predicted'] - T['actual']
                T['error_pct'] = T['error'] / T['actual'] * 100
                clf['scores'] = T
                # clf['scores'] = (T['predicted'] - T['actual']) / T['predicted'] * 100
                # clf['score'] = clf['scores'].mean()
                # print(clf['score'].round(1))
                # clf['scores'].round(1).disp(100)
                clf['output'] = y
                clf['model'] = model
                clf['scores'].disp(1000)

                # A = y.groupby('pred_code').sum().assign(error_pct=lambda x: (x['predicted']-x['actual'])/x['predicted']*100)
                
                # A['f1'] = y.groupby('pred_code').apply(lambda x: 1-f1_score(x['actual'], x['predicted']))
                # A['acc'] = y.groupby('pred_code').apply(lambda x: 1-accuracy_score(x['actual'], x['predicted']))
                # A['bal_acc'] = y.groupby('pred_code').apply(lambda x: 1-balanced_accuracy_score(x['actual'], x['predicted']))


                # automl_settings = {
                #     'time_budget':30,
                #     'label': targ,
                #     'task':"classification",
                #     # 'metric':'f1',
                #     'eval_method':'cv',
                #     'n_splits':3,
                #     'seed':42,
                #     'verbose':0,
                # }
                # model = AutoML(**automl_settings)
                # with warnings.catch_warnings(action='ignore'):
                #     # model.fit(dataframe=Z_train, **automl_settings)
                #     model.fit(x, y, **automl_settings)
                # print(model.get_params()['metric'], model.best_loss, model.best_estimator)#, model.best_config)
                # # predicted = model.predict(Z)
                # predicted = model.predict(X).prep
                # clf['score'] = model.best_loss
                # clf['model'] = model
                # clf['output'] = Y.to_frame().assign(predicted=predicted).addlevel('crse_code', path['crse_code']).addlevel('train_code', path['train_code'])
            # else:
            #     predicted = Y.copy()
            #     predicted[:] = False
            # rslt['output'] = Y.to_frame().assign(predicted=predicted).addlevel('crse_code', path['crse_code']).addlevel('train_code', path['train_code'])
            return clf
        self.run(grid, func)


    def get_optimal(self):
        grid = {'nm':'optimal', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes}
        def func(path):
            C = self.get(path | {'nm':'predicted'})
            if C is not None:
                E = [clf for trf in C.values() for imp in trf.values() for clf in imp.values()]
                return min(E, key=lambda clf: clf['score'])
            else:
                return dict()
        self.run(grid, func)


    def get_details(self):
        grid = {'nm':'details', 'styp_code':'all'}
        def func(path):
            A = pd.concat([C['output'] for S in self['optimal'].values() for T in S.values() for C in T.values() if 'output' in C and C['score'] < 1])
            return A
        self.run(grid, func)


    def summarize(self, D):
        grp = ['crse_code','levl_code','styp_code','pred_code','train_code','imp','sim']
        agg = lambda X: X.groupby(grp).apply(lambda x: pd.Series({
            'predicted': x['predicted'].sum(),
            'actual': x['actual'].sum(),
            # 'acc_pct': accuracy_score(x['actual'], x['predicted']) * 100,
            # 'bal_acc_pct': balanced_accuracy_score(x['actual'], x['predicted']) * 100,
            'f1': 1-f1_score(x['actual'], x['predicted'], zero_division=np.nan),
        }), include_groups=False)
        P = [D.reset_index(grp).reset_index(drop=True)]
        P = [q for p in P for q in [p, p.assign(styp_code='all')]]
        P = [q for p in P for q in [p, p.query(f"pred_code!={self.pred_code}").assign(train_code='all')]]
        with warnings.catch_warnings(action='ignore'):
            S = pd.concat([agg(p).join(self.mlt['all']) for p in P])
        mask = S.eval(f"train_code=='all'")
        S.loc[mask, ['predicted','actual']] *= (1.0 / len(self.term_codes))
        for k in ['predicted','actual']:
            S[k] *= S[k+'_mlt']
        S.insert(2, 'error', S['predicted'] - S['actual'])
        S.insert(3, 'error_pct', S['error'] / S['actual'] * 100)
        mask = S.eval(f"actual==0 or pred_code=={self.pred_code}")# or pred_code==train_code")
        S.loc[mask, 'actual':] = pd.NA
        S = S.reset_index()
        S['levl_desc'] = S['levl_code'].map({'ug':'undergraduate', 'g':'graduate'})
        S['styp_desc'] = S['styp_code'].map({'n':'new first time', 't':'transfer', 'r':'returning'})
        for k in ['pred','train','mlt']:
            S[k+'_desc'] = 'Fall ' + S[k+'_code'].astype('string').str[:4]
        return S.set_index(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','sim'])

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
passpwr = ['passthrough', pwrtrf]

# passdrop = passthru
# passpwr = passthru

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
    # 'cycle_day': 162,
    'crse_codes': [
        'agec2317',
        'ansc1119',
        'ansc1319',
        'anth2302',
        'anth2351',
        # 'arts1301',
        # 'arts1303',
        # 'arts1304',
        # 'arts3331',
        # 'biol1305',
        # 'biol1406',
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
        # 'datasets': 1,# 'iterations': 1, 'tune': False,
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
        # 'raw_df': True,
        # 'reg_df': True,
        # 'X': True,
        # 'Y': True,
        # 'mlt': True,
        ## 'inputs': True,
        # 'transformed': True,
        # 'imputed': True,
        # 'predicted': True,
        # 'optimal': True,
        # 'details': True,
        # 'summary': True,
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
        self.get_terms()
        self.get_raw_df()
        self.get_reg_df()
        self.get_X()
        self.get_Y()
        self.get_mlt()
        self.get_transformed()
        self.get_imputed()
        self.get_predicted()
        self.get_optimal()
        self.get_details()
        self.get_summary()
        # self.get_params()
        # self.push()