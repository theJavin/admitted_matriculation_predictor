test
import os, sys, time, datetime, pathlib, contextlib, dotenv, shutil, warnings, itertools as it
import pickle, dataclasses, typing, collections, oracledb
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.core.display import HTML
dotenv.load_dotenv()
C = ","
N = "\n"
T = "    "

def delete(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

def mkdir(path, overwrite=False):
    if overwrite:
        delete(path)
    path.mkdir(exist_ok=True, parents=True)

def write(fn, obj, overwrite=False, **kwargs):
    fn = pathlib.Path(fn)
    suf = ['.parq','.parquet','.pkl','.csv']
    assert fn.suffix in suf, f'Unknown suffix {fn.suffix} - must be one of {suf}'
    if overwrite:
        fn.unlink(missing_ok=True)
    if not fn.is_file():
        mkdir(fn.parent)
        if fn.suffix == '.pkl':
            with open(fn, 'wb') as f:
                pickle.dump(obj, f, **kwargs)
        else:
            obj = pd.DataFrame(obj).prep()
            if fn.suffix in ['.parq','.parquet']:
                obj.to_parquet(fn, **kwargs)
            elif fn.suffix == '.csv':
                obj.to_csv(fn, **kwargs)
    return obj

def read(fn, overwrite=False, **kwargs):
    fn = pathlib.Path(fn)
    suf = ['.parq','.parquet','.pkl','.csv']
    assert fn.suffix in suf, f'Unknown suffix {fn.suffix} - must be one of {suf}'
    if overwrite:
        fn.unlink(missing_ok=True)
    try:
        with open(fn, 'rb') as f:
            return pickle.load(f, **kwargs)
    except:
        try:
            return pd.read_parquet(fn, **kwargs).prep()
        except:
            try:
                return pd.read_csv(fn, **kwargs).prep()
            except:
                return None

def listify(X):
    if X is None or X is np.nan:
        return []
    elif isinstance(X, (str,int,float,bool)):
        return [X]
    else:
        return list(X)
    
def setify(X):
    return set(listify(X))

def uniquify(X):
    if isinstance(X, (list, tuple, set)):
        return type(X)(dict.fromkeys(listify(X)))
    else:
        return X

def sort(X):
    if isinstance(X, (list, tuple, set)):
        return type(X)(sorted(X))
    elif isinstance(X, dict):
        return dict(sorted(X.items()))
    else:
        return X

def instantiate(x):
    try:
        return x()
    except:
        return x

def rjust(x, width, fillchar=' '):
    return str(x).rjust(width,str(fillchar))

def ljust(x, width, fillchar=' '):
    return str(x).ljust(width,str(fillchar))

def join(x, sep=', ', pre='', post=''):
    return f"{pre}{str(sep).join(map(str,listify(x)))}{post}"

def indent(qry, lev=1):
    return (N+qry.strip()).replace(N,N+T*lev) if lev>0 else qry

def isnum(x):
    return str(x).isnumeric()

def isqry(qry):
    return 'select' in qry.lower()

def mkqry(qry):
    return qry.strip() if isqry(qry) else f"select * from {qry}"

def subqry(qry, lev=1):
    NT = N+T*(lev-1)
    return "(" + (N+qry.strip()).replace(N,NT+T) + NT + ")" if lev>0 else qry

### pandas helpers ###
def pd_ext(func):
    def wrapper(X, *args, **kwargs):
        try:
            Y = func(X, *args, **kwargs)
        except:
            Y = pd.DataFrame(X)
            try:
                Y = func(Y, *args, **kwargs)
            except:
                Y = Y.apply(func, *args, **kwargs)
        if isinstance(X, pd.Series):
            try:
                Y = Y.squeeze()
            except:
                pass
        return Y
    wrapper.__name__ = func.__name__
    return wrapper

@pd_ext
def disp(df, max_rows=4, max_cols=200, **kwargs):
    display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))

@pd_ext
def to_numeric(ser, dtype_backend='numpy_nullable', errors='ignore'):
    assert isinstance(ser, pd.Series)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=(FutureWarning,UserWarning))
        dt = str(ser.dtype).lower()
        if 'geometry' not in dt and 'bool' not in dt and 'category' not in dt:
            ser = pd.to_numeric(ser.astype('string').str.lower().str.strip().replace('',pd.NA), downcast='integer', errors=errors)
            if pd.api.types.is_string_dtype(ser):
                ser = pd.to_datetime(ser, errors=errors)
            elif pd.api.types.is_integer_dtype(ser):
                ser = ser.astype('Int64', errors=errors)
        return ser.convert_dtypes(dtype_backend)

@pd_ext
def prep(X, cap="casefold"):
    if isinstance(X, str):
        if cap:
            X = getattr(X, cap)()
        return X.strip()
    elif isinstance(X, (list, tuple, set, np.ndarray, pd.Index)):
        return type(X)((prep(x, cap) for x in X))
    elif isinstance(X, dict):
        return {prep(k,cap):prep(v,cap) for k,v in X.items()}
    elif isinstance(X, pd.DataFrame):
        rename_column = lambda x: prep(x, cap).replace(' ','_').replace('-','_') if isinstance(x, str) else x
        X = X.rename(columns=rename_column).rename_axis(index=rename_column)
        idx = pd.MultiIndex.from_frame(X[[]].reset_index().to_numeric())
        return X.to_numeric().set_index(idx).rename_axis(X.index.names)
    else:
        raise Exception(f'prep undefined for {type(X)}')

@pd_ext
def categorize(ser):
    assert isinstance(ser, pd.Series)
    return ser.astype('category') if pd.api.types.is_string_dtype(ser) else ser

@pd_ext
def binarize(ser):
    assert isinstance(ser, pd.Series)
    s = set(ser.dropna())
    if s:
        if s.issubset({'y','Y'}):
            ser = ser.notnull().astype('boolean')
        elif s.issubset({0,1}):
            ser = ser.astype('boolean')
    return ser

@pd_ext
def rnd(ser, digits=0):
    assert isinstance(ser, pd.Series)
    return ser.round(digits).prep()

@pd_ext
def vc(df, by, dropna=False, **kwargs):
    return df.groupby(by, dropna=dropna, **kwargs).size().to_frame('ct').assign(pct=lambda x: (x['ct']/df.shape[0]*100).rnd(digits))

@pd_ext
def missing(df, digits=1):
    return df.isnull().sum().sort_values(ascending=False).to_frame('ct').query('ct>0').assign(pct=lambda x: (x['ct']/df.shape[0]*100).rnd(digits))

@pd_ext
def impute(df, col, val=None, grp=None):
    val = val if val is not None else 'median' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
    if val in ['median']:
        func = lambda x: x.median()
    elif val in ['mean','ave','avg','average']:
        func = lambda x: x.mean()
    elif val in ['mode','most_frequent']:
        func = lambda x: x.mode()[0]
    elif val in ['max']:
        func = lambda x: x.max()
    elif val in ['min']:
        func = lambda x: x.min()
    else:
        func = lambda x: val
    return (df if grp is None else df.groupby(grp))[col].transform(lambda x: x.fillna(func(x)))

@pd_ext
def unmelt(self, level=-1, names=None):
    df = self.unstack(level).droplevel(0,1).rename_axis(columns=None)
    if names is not None:
        df.columns = listify(names)
    return df

for func in [disp, to_numeric, prep, categorize, binarize, rnd, vc, missing, impute, unmelt]:
    for cls in [pd.DataFrame, pd.Series]:
        setattr(cls, func.__name__, func)

# def missing(df):
#     m = pd.DataFrame(df).isnull().sum().sort_values(ascending=False).to_frame('ct').query('ct>0')
#     m['pct'] = (m['ct'] / df.shape[0] * 100).round(1)
#     return m
# pd.DataFrame.missing = missing
# pd.Series.missing = missing

# # def to_numeric(ser, errors='ignore'):
# #     dt = str(ser.dtype).lower()
# #     if 'geometry' not in dt and 'bool' not in dt:
# #         with warnings.catch_warnings(action='ignore'):
# #             ser = pd.to_numeric(ser.astype('string').str.lower().str.strip().replace('',pd.NA),
# #                                 errors=errors, downcast='integer', dtype_backend=DTYPE_BACKEND)
# #             if pd.api.types.is_string_dtype(ser):
# #                 ser = pd.to_datetime(ser, errors='ignore')
# #             elif pd.api.types.is_integer_dtype(ser):
# #                 ser = ser.astype('Int64', errors='ignore')
# #     return ser.convert_dtypes(DTYPE_BACKEND)

# def binarize(df):
#     def f(ser):
#         s = set(ser.dropna())
#         if len(s) == 0:
#             return ser
#         if s.issubset({'y','Y'}):
#             return ser.notnull().astype('boolean')
#         elif s.issubset({0,1}):
#             return ser.astype('boolean')
#         else:
#             return ser
#     return df.apply(f)
# pd.DataFrame.binarize = binarize

# def categorize(df):
#     def f(ser):
#         if pd.api.types.is_string_dtype(ser):
#             return ser.astype('category')
#         else:
#             return ser
#     return df.apply(f)
# pd.DataFrame.categorize = categorize

# def to_numeric(df, errors='ignore'):
#     with warnings.catch_warnings(action='ignore'):
#         def f(ser):
#             dt = str(ser.dtype).lower()
#             if 'geometry' not in dt and 'bool' not in dt and 'category' not in dt:
#                 ser = pd.to_numeric(ser.astype('string').str.lower().str.strip().replace('',pd.NA), errors=errors, downcast='integer', dtype_backend=DTYPE_BACKEND)
#                 if pd.api.types.is_string_dtype(ser):
#                     ser = pd.to_datetime(ser, errors='ignore')
#                 elif pd.api.types.is_integer_dtype(ser):
#                     ser = ser.astype('Int64', errors='ignore')
#             return ser
#         return df.apply(f).convert_dtypes(DTYPE_BACKEND)
# pd.DataFrame.to_numeric = to_numeric

# def unmelt(self, level=-1, names=None):
#     df = self.unstack(level).droplevel(0,1).rename_axis(columns=None)
#     if names is not None:
#         df.columns = listify(names)
#     return df
# pd.DataFrame.unmelt = unmelt

# def rindex(df, keys=None, keep=True, **kwargs):
#     df = pd.DataFrame(df)
#     keys = listify(keys)
#     idx = pd.Index(df.index.names).drop(None, errors='ignore')
#     rst = list(idx if keep else idx.intersection(keys))
#     df = df.drop(columns=rst, errors='ignore').reset_index(rst, drop=False).reset_index(drop=True)
#     return df.set_index(keys, **kwargs) if keys else df
# pd.Series.rindex = rindex
# pd.DataFrame.rindex = rindex

# def grpby(df, by, keep=False, dropna=False, **kwargs):
#     return df.rindex(by, keep).groupby(by, dropna=dropna, **kwargs)
# pd.Series.grpby = grpby
# pd.DataFrame.grpby = grpby

# def vc(df, by, dropna=False, **kwargs):
#     return df.grpby(by, dropna=dropna, **kwargs).size().to_frame('ct')
# pd.Series.vc = vc
# pd.DataFrame.vc = vc

# def impute(df, col, val=None, grp=None):
#     val = val if val is not None else 'median' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
#     if val in ['median']:
#         func = lambda x: x.median()
#     elif val in ['mean','ave','avg','average']:
#         func = lambda x: x.mean()
#     elif val in ['mode','most_frequent']:
#         func = lambda x: x.mode()[0]
#     elif val in ['max']:
#         func = lambda x: x.max()
#     elif val in ['min']:
#         func = lambda x: x.min()
#     else:
#         func = lambda x: val
#     return (df if grp is None else df.groupby(grp))[col].transform(lambda x: x.fillna(func(x)))
# pd.DataFrame.impute = impute

def pctl(p):
    p = round(p if p >= 1 else p*100)
    f = lambda x: x.quantile(p/100)
    f.__name__ = f'{p}%'
    return f

def IQR(x):
    return pctl(75)(x)-pctl(25)(x)

def ran(x):
    return pctl(100)(x)-pctl(0)(x)

summary = ['count',pctl(0),pctl(25),pctl(50),pctl(75),pctl(100),IQR,ran]




class MyBaseClass():
    def __contains__(self, key):
        return key in self.__dict__
    def __getitem__(self, key):
        return self.__dict__[key]
    def __delitem__(self, key):
        if key in self:
            del self.__dict__[key]
    def __setitem__(self, key, val):
        self.__dict__[key] = val

@dataclasses.dataclass
class Oracle(MyBaseClass):
    database: str = 'WFOCUST'
    timeout_default: int = 60000

    def execute(self, qry, show=False, timeout=None, **opts):
        qry = mkqry(qry)
        if show:
            print(qry)
        with warnings.catch_warnings(action='ignore'), oracledb.connect(user=os.environ.get(self.database+'_USER'), password=os.environ.get(self.database+'_PASS'), dsn=os.environ.get(self.database+'_DSN')) as connection:
            connection.call_timeout = self.timeout_default if timeout is None else timeout
            try:
                n_ses = pd.read_sql(f"select count(*) from sys.v_$session where username='{os.environ.get(self.database+'_USER')}'", connection).squeeze()
                print(f"{n_ses} active sessions")
            except:
                pass
            df = pd.read_sql(qry, connection).prep()
        return df
    
    def head(self, qry, rows=2, **opts):
        qry = mkqry(qry) + f"\nfetch first {rows} rows only"
        return self.execute(qry, **opts)

    def shape(self, qry, **opts):
        qry = f"select count(*) from {subqry(qry)}"
        m = self.execute(qry, **opts).squeeze()
        print(m)
        return m
    
    def value_counts(self, qry, val, **opts):
        sub = subqry(qry) if isqry(qry) else qry
        val = join(val)
        qry = f"select {val}, count(*) as ct from {subqry(qry)} group by {val} order by {val}"
        return self.execute(qry, **opts)

    def repeats(self, qry, val, **opts):
        sub = subqry(qry) if isqry(qry) else qry
        val = join(val)
        qry = f"select ct, count(*) as rep from (select count(*) as ct from {subqry(qry)} group by {val}) group by ct order by ct"
        return self.execute(qry, **opts)

    def dtypes(self, qry):
        dt = self.execute(SQL(qry).qry().head(10)).dtypes
        dt.display()
        return dt