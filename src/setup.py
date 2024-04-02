import os, sys, time, datetime, pathlib, contextlib, dotenv, shutil, warnings, itertools as it
import pickle, joblib, json, dataclasses, typing, collections, oracledb
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from copy import deepcopy
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
dotenv.load_dotenv()
C = ","
N = "\n"
T = "    "

def listify(X, sort=False, **kwargs):
    if X is None or X is np.nan:
        X = []
    elif isinstance(X, (str,int,float,bool)) or callable(X):
        X = [X]
    else:
        X = list(X)
    return sorted(X, **kwargs) if sort else X

def setify(X):
    return set(listify(X))

def mysort(X, **kwargs):
    if isinstance(X, dict):
        return dict(sorted(X.items(), **kwargs))
    else:
        return listify(X, True, **kwargs)
        # return sorted(listify(X), **kwargs)
    
def uniquify(X, sort=True, **kwargs):
    if not isinstance(X, dict):
        X = listify(dict.fromkeys(listify(X)))
    return mysort(X, **kwargs) if sort else X

def subdct(X, keys=None, sort=False, **kwargs):
    X = X if keys is None else {k: X[k] for k in listify(keys)}
    return mysort(X, **kwargs) if sort else X

# def pop(x, key):
#     y = x.copy()
#     y.pop(key)
#     return y

def cartesian(dct, sort=True):
    """Creates the Cartesian product of a dictionary with list-like values"""
    try:
        D = {key: listify(val) for key, val in dct.items()}
        D = [dict(zip(D.keys(), x)) for x in it.product(*D.values())]
        return [mysort(x) for x in D] if sort else D
    except:
        return dict()

def nest(path, dct=dict(), val=None):
    path = listify(path.values() if isinstance(path, dict) else path)
    k = path.pop(-1)
    a = dct
    for p in path:
        a.setdefault(p, dict())
        a = a[p]
    if val is None:
        return a[k]
    else:
        a[k] = val
        return dct

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
        return Y
    wrapper.__name__ = func.__name__
    for cls in [pd.DataFrame, pd.Series]:
        setattr(cls, wrapper.__name__, wrapper)
    return wrapper

@pd_ext
def disp(df, max_rows=4, max_cols=200, **kwargs):
    display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
    # print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))


@pd_ext
def prep_number(ser, dtype_backend='numpy_nullable'):
    assert isinstance(ser, pd.Series)
    if pd.api.types.is_string_dtype(ser) or pd.api.types.is_object_dtype(ser):
        ser = ser.astype('string')
        try:
            ser = pd.to_datetime(ser)
        except ValueError:
            try:
                ser = pd.to_numeric(ser, downcast='integer')
            except ValueError:
                pass
    return ser.astype('Int64') if pd.api.types.is_integer_dtype(ser) else ser.convert_dtypes(dtype_backend)

@pd_ext
def prep_string(ser, cap="casefold"):
    assert isinstance(ser, pd.Series)
    ser = ser.prep_number()
    return getattr(ser.str, cap)().replace('',pd.NA) if pd.api.types.is_string_dtype(ser) else ser

@pd_ext
def prep_category(ser):
    assert isinstance(ser, pd.Series)
    ser = ser.prep_string()
    return ser.astype('category') if pd.api.types.is_string_dtype(ser) else ser

@pd_ext
def prep_bool(ser):
    assert isinstance(ser, pd.Series)
    ser = ser.prep_string()
    vals = ser.dropna().unique()
    if len(vals) in [1, 2]:
        vals = set(vals)
        for s in [['false','true'], ['n','y'], [0, 1]]:
            if vals.issubset(s):
                ser = (ser == s[1]).astype('boolean').fillna(False)
    return ser

@pd_ext
def prep(X, cap='casefold'):
    if isinstance(X, str):
        if cap:
            X = getattr(X, cap)()
        return X.strip()
    elif isinstance(X, (list, tuple, set, np.ndarray, pd.Index)):
        return type(X)((prep(x, cap) for x in X))
    elif isinstance(X, dict):
        return {prep(k,cap):prep(v,cap) for k,v in X.items()}
    elif isinstance(X, pd.DataFrame):
        g = lambda x: prep(x, cap).replace(' ','_').replace('-','_') if isinstance(x, str) else x
        X = X.rename(columns=g).rename_axis(index=g)
        idx = pd.MultiIndex.from_frame(X[[]].reset_index().prep_string())
        return X.prep_string().set_index(idx).rename_axis(X.index.names)
    elif isinstance(X, pd.Series):
        assert 1==2
    else:
        return X

@pd_ext
def addlevel(df, level, val):
    return df.assign(**{level:val}).set_index(level, append=True)

@pd_ext
def rnd(ser, decimals=0):
    assert isinstance(ser, pd.Series)
    return ser.round(decimals=decimals).prep()

@pd_ext
def vc(df, by, dropna=False, digits=1, **kwargs):
    return df.groupby(by, dropna=dropna, observed=False, **kwargs).size().to_frame('ct').assign(pct=lambda x: (x['ct']/df.shape[0]*100).rnd(digits))

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

# @pd_ext
# def unmelt(self, level=-1, names=None):
#     df = self.unstack(level).droplevel(0,1).rename_axis(columns=None)
#     if names is not None:
#         df.columns = listify(names)
#     return df

def delete(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

def mkdir(path, overwrite=False):
    if overwrite:
        delete(path)
    path.mkdir(exist_ok=True, parents=True)

def write(fn, obj, overwrite=False, protocol=5, **kwargs):
    fn = pathlib.Path(fn)
    suf = ['.parq','.parquet','.pkl','.csv']
    assert fn.suffix in suf, f'Unknown suffix {fn.suffix} - must be one of {suf}'
    if overwrite:
        fn.unlink(missing_ok=True)
    if not fn.is_file():
        mkdir(fn.parent)
        if fn.suffix == '.pkl':
            with open(fn, 'wb') as f:
                # joblib.dump(obj, f, **kwargs)
                pickle.dump(obj, f, protocol=protocol, **kwargs)
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
            # return joblib.load(f, **kwargs)
            return pickle.load(f, **kwargs)
    except:
        try:
            return pd.read_parquet(fn, **kwargs).prep()
        except:
            try:
                return pd.read_csv(fn, **kwargs).prep()
            except:
                return None

def pctl(p):
    p = round(p if p >= 1 else p*100)
    f = lambda x: x.quantile(p/100)
    f.__name__ = f'{p}%'.rjust(4)
    return f

def IQR(x):
    return pctl(75)(x)-pctl(25)(x)

def ran(x):
    return pctl(100)(x)-pctl(0)(x)

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