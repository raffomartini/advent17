#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xfb8faa05

# Compiled with Coconut version 1.3.1 [Dead Parrot]

'''
--- Part Two ---

You notice a progress bar that jumps to 50% completion. Apparently, the door isn't yet satisfied, but it did emit a star as encouragement. The instructions change:

Now, instead of considering the next digit, it wants you to consider the digit halfway around the circular list. That is, if your list contains 10 items, only include a digit in your sum if the digit 10/2 = 5 steps forward matches it. Fortunately, your list has an even number of elements.

For example:

'1212' produces 6: the list contains 4 items, and all four digits match the digit 2 items ahead.
'1221' produces 0, because every comparison is between a 1 and a 2.
'123425' produces 4, because both 2s match each other, but no other digit has a match.
'123123' produces 12.
'12131415' produces 4.
'''

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys
if _coconut_sys.version_info < (3,):
    py_chr, py_filter, py_hex, py_input, py_int, py_map, py_object, py_oct, py_open, py_print, py_range, py_str, py_zip, py_filter, py_reversed, py_enumerate = chr, filter, hex, input, int, map, object, oct, open, print, range, str, zip, filter, reversed, enumerate
    py_raw_input, py_xrange = raw_input, xrange
    _coconut_NotImplemented, _coconut_raw_input, _coconut_xrange, _coconut_int, _coconut_long, _coconut_print, _coconut_str, _coconut_unicode, _coconut_repr = NotImplemented, raw_input, xrange, int, long, print, str, unicode, repr
    from future_builtins import *
    chr, str = unichr, unicode
    from io import open
    class object(object):
        __slots__ = ()
        def __ne__(self, other):
            eq = self == other
            if eq is _coconut_NotImplemented:
                return eq
            return not eq
    class int(_coconut_int):
        __slots__ = ()
        if hasattr(_coconut_int, "__doc__"):
            __doc__ = _coconut_int.__doc__
        class __metaclass__(type):
            def __instancecheck__(cls, inst):
                return _coconut.isinstance(inst, (_coconut_int, _coconut_long))
            def __subclasscheck__(cls, subcls):
                return _coconut.issubclass(subcls, (_coconut_int, _coconut_long))
    class range(object):
        __slots__ = ("_xrange",)
        if hasattr(_coconut_xrange, "__doc__"):
            __doc__ = _coconut_xrange.__doc__
        def __init__(self, *args):
            self._xrange = _coconut_xrange(*args)
        def __iter__(self):
            return _coconut.iter(self._xrange)
        def __reversed__(self):
            return _coconut.reversed(self._xrange)
        def __len__(self):
            return _coconut.len(self._xrange)
        def __contains__(self, elem):
            return elem in self._xrange
        def __getitem__(self, index):
            if _coconut.isinstance(index, _coconut.slice):
                args = _coconut.slice(*self._args)
                start, stop, step, ind_step = (args.start if args.start is not None else 0), args.stop, (args.step if args.step is not None else 1), (index.step if index.step is not None else 1)
                return self.__class__((start if ind_step >= 0 else stop - step) if index.start is None else start + step * index.start if index.start >= 0 else stop + step * index.start, (stop if ind_step >= 0 else start - step) if index.stop is None else start + step * index.stop if index.stop >= 0 else stop + step * index.stop, step if index.step is None else step * index.step)
            else:
                return self._xrange[index]
        def count(self, elem):
            """Count the number of times elem appears in the range."""
            return _coconut_int(elem in self._xrange)
        def index(self, elem):
            """Find the index of elem in the range."""
            if elem not in self._xrange: raise _coconut.ValueError(_coconut.repr(elem) + " is not in range")
            start, _, step = self._xrange.__reduce_ex__(2)[1]
            return (elem - start) // step
        def __repr__(self):
            return _coconut.repr(self._xrange)[1:]
        @property
        def _args(self):
            return self._xrange.__reduce__()[1]
        def __reduce_ex__(self, protocol):
            return (self.__class__, self._xrange.__reduce_ex__(protocol)[1])
        def __reduce__(self):
            return self.__reduce_ex__(_coconut.pickle.DEFAULT_PROTOCOL)
        def __hash__(self):
            return _coconut.hash(self._args)
        def __copy__(self):
            return self.__class__(*self._args)
        def __eq__(self, other):
            return _coconut.isinstance(other, self.__class__) and self._args == other._args
    from collections import Sequence as _coconut_Sequence
    _coconut_Sequence.register(range)
    from functools import wraps as _coconut_wraps
    @_coconut_wraps(_coconut_print)
    def print(*args, **kwargs):
        file = kwargs.get("file", _coconut_sys.stdout)
        flush = kwargs.get("flush", False)
        if "flush" in kwargs:
            del kwargs["flush"]
        if _coconut.hasattr(file, "encoding") and file.encoding is not None:
            _coconut_print(*(_coconut_unicode(x).encode(file.encoding) for x in args), **kwargs)
        else:
            _coconut_print(*(_coconut_unicode(x).encode() for x in args), **kwargs)
        if flush:
            file.flush()
    @_coconut_wraps(_coconut_raw_input)
    def input(*args, **kwargs):
        if _coconut.hasattr(_coconut_sys.stdout, "encoding") and _coconut_sys.stdout.encoding is not None:
            return _coconut_raw_input(*args, **kwargs).decode(_coconut_sys.stdout.encoding)
        return _coconut_raw_input(*args, **kwargs).decode()
    @_coconut_wraps(_coconut_repr)
    def repr(obj):
        if isinstance(obj, _coconut_unicode):
            return _coconut_unicode(_coconut_repr(obj)[1:])
        if isinstance(obj, _coconut_str):
            return "b" + _coconut_unicode(_coconut_repr(obj))
        return _coconut_unicode(_coconut_repr(obj))
    ascii = repr
    def raw_input(*args):
        """Coconut uses Python 3 "input" instead of Python 2 "raw_input"."""
        raise _coconut.NameError('Coconut uses Python 3 "input" instead of Python 2 "raw_input"')
    def xrange(*args):
        """Coconut uses Python 3 "range" instead of Python 2 "xrange"."""
        raise _coconut.NameError('Coconut uses Python 3 "range" instead of Python 2 "xrange"')
    if _coconut_sys.version_info < (2, 7):
        import functools as _coconut_functools, copy_reg as _coconut_copy_reg
        def _coconut_new_partial(func, args, keywords):
            return _coconut_functools.partial(func, *(args if args is not None else ()), **(keywords if keywords is not None else {}))
        _coconut_copy_reg.constructor(_coconut_new_partial)
        def _coconut_reduce_partial(self):
            return (_coconut_new_partial, (self.func, self.args, self.keywords))
        _coconut_copy_reg.pickle(_coconut_functools.partial, _coconut_reduce_partial)
else:
    py_chr, py_filter, py_hex, py_input, py_int, py_map, py_object, py_oct, py_open, py_print, py_range, py_str, py_zip, py_filter, py_reversed, py_enumerate = chr, filter, hex, input, int, map, object, oct, open, print, range, str, zip, filter, reversed, enumerate
class _coconut(object):
    import collections, copy, functools, imp, itertools, operator, types, weakref
    if _coconut_sys.version_info < (3,):
        import cPickle as pickle
    else:
        import pickle
    if _coconut_sys.version_info >= (2, 7):
        OrderedDict = collections.OrderedDict
    else:
        OrderedDict = dict
    if _coconut_sys.version_info < (3, 3):
        abc = collections
    else:
        import collections.abc as abc
    Exception, IndexError, KeyError, NameError, TypeError, ValueError, StopIteration, classmethod, dict, enumerate, filter, frozenset, getattr, hasattr, hash, id, int, isinstance, issubclass, iter, len, list, map, min, max, next, object, property, range, reversed, set, slice, str, sum, super, tuple, zip, repr, bytearray = Exception, IndexError, KeyError, NameError, TypeError, ValueError, StopIteration, classmethod, dict, enumerate, filter, frozenset, getattr, hasattr, hash, id, int, isinstance, issubclass, iter, len, list, map, min, max, next, object, property, range, reversed, set, slice, str, sum, super, tuple, zip, staticmethod(repr), bytearray
def _coconut_NamedTuple(name, fields):
    return _coconut.collections.namedtuple(name, [x for x, t in fields])
class MatchError(Exception):
    """Pattern-matching error. Has attributes .pattern and .value."""
    __slots__ = ("pattern", "value")
class _coconut_tail_call(object):
    __slots__ = ("func", "args", "kwargs")
    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs
_coconut_tco_func_dict = {}
def _coconut_tco(func):
    @_coconut.functools.wraps(func)
    def tail_call_optimized_func(*args, **kwargs):
        call_func = func
        while True:
            wkref = _coconut_tco_func_dict.get(_coconut.id(call_func))
            if wkref is not None and wkref() is call_func:
                call_func = call_func._coconut_tco_func
            result = call_func(*args, **kwargs)  # pass --no-tco to clean up your traceback
            if not isinstance(result, _coconut_tail_call):
                return result
            call_func, args, kwargs = result.func, result.args, result.kwargs
    tail_call_optimized_func._coconut_tco_func = func
    _coconut_tco_func_dict[_coconut.id(tail_call_optimized_func)] = _coconut.weakref.ref(tail_call_optimized_func)
    return tail_call_optimized_func
def _coconut_igetitem(iterable, index):
    if isinstance(iterable, (_coconut_reversed, _coconut_map, _coconut.filter, _coconut.zip, _coconut_enumerate, _coconut_count, _coconut.abc.Sequence)):
        return iterable[index]
    if not _coconut.isinstance(index, _coconut.slice):
        if index < 0:
            return _coconut.collections.deque(iterable, maxlen=-index)[0]
        return _coconut.next(_coconut.itertools.islice(iterable, index, index + 1))
    if index.start is not None and index.start < 0 and (index.stop is None or index.stop < 0) and index.step is None:
        queue = _coconut.collections.deque(iterable, maxlen=-index.start)
        if index.stop is not None:
            queue = _coconut.tuple(queue)[:index.stop - index.start]
        return queue
    if (index.start is not None and index.start < 0) or (index.stop is not None and index.stop < 0) or (index.step is not None and index.step < 0):
        return _coconut.tuple(iterable)[index]
    return _coconut.itertools.islice(iterable, index.start, index.stop, index.step)
class _coconut_base_compose(object):
    __slots__ = ("func", "funcstars")
    def __init__(self, func, *funcstars):
        self.func = func
        self.funcstars = []
        for f, star in funcstars:
            if isinstance(f, _coconut_base_compose):
                self.funcstars.append((f.func, star))
                self.funcstars += f.funcstars
            else:
                self.funcstars.append((f, star))
    def __call__(self, *args, **kwargs):
        arg = self.func(*args, **kwargs)
        for f, star in self.funcstars:
            arg = f(*arg) if star else f(arg)
        return arg
    def __repr__(self):
        return _coconut.repr(self.func) + " " + " ".join(("..*> " if star else "..> ") + _coconut.repr(f) for f, star in self.funcstars)
    def __reduce__(self):
        return (self.__class__, (self.func,) + _coconut.tuple(self.funcstars))
def _coconut_forward_compose(func, *funcs): return _coconut_base_compose(func, *((f, False) for f in funcs))
def _coconut_back_compose(*funcs): return _coconut_forward_compose(*_coconut.reversed(funcs))
def _coconut_forward_star_compose(func, *funcs): return _coconut_base_compose(func, *((f, True) for f in funcs))
def _coconut_back_star_compose(*funcs): return _coconut_forward_star_compose(*_coconut.reversed(funcs))
def _coconut_pipe(x, f): return f(x)
def _coconut_star_pipe(xs, f): return f(*xs)
def _coconut_back_pipe(f, x): return f(x)
def _coconut_back_star_pipe(f, xs): return f(*xs)
def _coconut_bool_and(a, b): return a and b
def _coconut_bool_or(a, b): return a or b
def _coconut_none_coalesce(a, b): return a if a is not None else b
def _coconut_minus(a, *rest):
    if not rest:
        return -a
    for b in rest:
        a = a - b
    return a
@_coconut.functools.wraps(_coconut.itertools.tee)
def tee(iterable, n=2):
    if n >= 0 and _coconut.isinstance(iterable, (_coconut.tuple, _coconut.frozenset)):
        return (iterable,) * n
    if n > 0 and (_coconut.hasattr(iterable, "__copy__") or _coconut.isinstance(iterable, _coconut.abc.Sequence)):
        return (iterable,) + _coconut.tuple(_coconut.copy.copy(iterable) for _ in _coconut.range(n - 1))
    return _coconut.itertools.tee(iterable, n)
class reiterable(object):
    """Allows an iterator to be iterated over multiple times."""
    __slots__ = ("iter",)
    def __init__(self, iterable):
        self.iter = iterable
    def __iter__(self):
        self.iter, out = _coconut_tee(self.iter)
        return _coconut.iter(out)
    def __getitem__(self, index):
        return _coconut_igetitem(_coconut.iter(self), index)
    def __reversed__(self):
        return _coconut_reversed(_coconut.iter(self))
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "reiterable(" + _coconut.repr(self.iter) + ")"
    def __reduce__(self):
        return (self.__class__, (self.iter,))
    def __copy__(self):
        return self.__class__(_coconut.copy.copy(self.iter))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class scan(object):
    """Reduce func over iterable, yielding intermediate results."""
    __slots__ = ("func", "iter")
    def __init__(self, func, iterable):
        self.func, self.iter = func, iterable
    def __iter__(self):
        acc = empty_acc = _coconut.object()
        for item in self.iter:
            if acc is empty_acc:
                acc = item
            else:
                acc = self.func(acc, item)
            yield acc
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "scan(" + _coconut.repr(self.iter) + ")"
    def __reduce__(self):
        return (self.__class__, (self.func, self.iter))
    def __copy__(self):
        return self.__class__(self.func, _coconut.copy.copy(self.iter))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class reversed(object):
    __slots__ = ("_iter",)
    if hasattr(_coconut.map, "__doc__"):
        __doc__ = _coconut.reversed.__doc__
    def __new__(cls, iterable):
        if _coconut.isinstance(iterable, _coconut.range):
            return iterable[::-1]
        if not _coconut.hasattr(iterable, "__reversed__") or _coconut.isinstance(iterable, (_coconut.list, _coconut.tuple)):
            return _coconut.object.__new__(cls)
        return _coconut.reversed(iterable)
    def __init__(self, iterable):
        self._iter = iterable
    def __iter__(self):
        return _coconut.iter(_coconut.reversed(self._iter))
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return _coconut_igetitem(self._iter, _coconut.slice(-(index.start + 1) if index.start is not None else None, -(index.stop + 1) if index.stop else None, -(index.step if index.step is not None else 1)))
        return _coconut_igetitem(self._iter, -(index + 1))
    def __reversed__(self):
        return self._iter
    def __len__(self):
        return _coconut.len(self._iter)
    def __repr__(self):
        return "reversed(" + _coconut.repr(self._iter) + ")"
    def __hash__(self):
        return -_coconut.hash(self._iter)
    def __reduce__(self):
        return (self.__class__, (self._iter,))
    def __copy__(self):
        return self.__class__(_coconut.copy.copy(self._iter))
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._iter == other._iter
    def __contains__(self, elem):
        return elem in self._iter
    def count(self, elem):
        """Count the number of times elem appears in the reversed iterator."""
        return self._iter.count(elem)
    def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1
    def __fmap__(self, func):
        return self.__class__(_coconut_map(func, self._iter))
class map(_coconut.map):
    __slots__ = ("_func", "_iters")
    if hasattr(_coconut.map, "__doc__"):
        __doc__ = _coconut.map.__doc__
    def __new__(cls, function, *iterables):
        new_map = _coconut.map.__new__(cls, function, *iterables)
        new_map._func, new_map._iters = function, iterables
        return new_map
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(self._func, *(_coconut_igetitem(i, index) for i in self._iters))
        return self._func(*(_coconut_igetitem(i, index) for i in self._iters))
    def __reversed__(self):
        return self.__class__(self._func, *(_coconut_reversed(i) for i in self._iters))
    def __len__(self):
        return _coconut.min(_coconut.len(i) for i in self._iters)
    def __repr__(self):
        return "map(" + _coconut.repr(self._func) + ", " + ", ".join((_coconut.repr(i) for i in self._iters)) + ")"
    def __reduce__(self):
        return (self.__class__, (self._func,) + self._iters)
    def __reduce_ex__(self, _):
        return self.__reduce__()
    def __copy__(self):
        return self.__class__(self._func, *_coconut.map(_coconut.copy.copy, self._iters))
    def __fmap__(self, func):
        return self.__class__(_coconut_forward_compose(self._func, func), *self._iters)
class parallel_map(map):
    """Multi-process implementation of map using concurrent.futures.
    Requires arguments to be pickleable."""
    __slots__ = ()
    def __iter__(self):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            return _coconut.iter(_coconut.tuple(executor.map(self._func, *self._iters)))
    def __repr__(self):
        return "parallel_" + _coconut_map.__repr__(self)
class concurrent_map(map):
    """Multi-thread implementation of map using concurrent.futures."""
    __slots__ = ()
    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor
        from multiprocessing import cpu_count  # cpu_count() * 5 is the default Python 3.5 thread count
        with ThreadPoolExecutor(cpu_count() * 5) as executor:
            return _coconut.iter(_coconut.tuple(executor.map(self._func, *self._iters)))
    def __repr__(self):
        return "concurrent_" + _coconut_map.__repr__(self)
class filter(_coconut.filter):
    __slots__ = ("_func", "_iter")
    if hasattr(_coconut.filter, "__doc__"):
        __doc__ = _coconut.filter.__doc__
    def __new__(cls, function, iterable):
        new_filter = _coconut.filter.__new__(cls, function, iterable)
        new_filter._func, new_filter._iter = function, iterable
        return new_filter
    def __reversed__(self):
        return self.__class__(self._func, _coconut_reversed(self._iter))
    def __repr__(self):
        return "filter(" + _coconut.repr(self._func) + ", " + _coconut.repr(self._iter) + ")"
    def __reduce__(self):
        return (self.__class__, (self._func, self._iter))
    def __reduce_ex__(self, _):
        return self.__reduce__()
    def __copy__(self):
        return self.__class__(self._func, _coconut.copy.copy(self._iter))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class zip(_coconut.zip):
    __slots__ = ("_iters",)
    if hasattr(_coconut.zip, "__doc__"):
        __doc__ = _coconut.zip.__doc__
    def __new__(cls, *iterables):
        new_zip = _coconut.zip.__new__(cls, *iterables)
        new_zip._iters = iterables
        return new_zip
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(*(_coconut_igetitem(i, index) for i in self._iters))
        return _coconut.tuple(_coconut_igetitem(i, index) for i in self._iters)
    def __reversed__(self):
        return self.__class__(*(_coconut_reversed(i) for i in self._iters))
    def __len__(self):
        return _coconut.min(_coconut.len(i) for i in self._iters)
    def __repr__(self):
        return "zip(" + ", ".join((_coconut.repr(i) for i in self._iters)) + ")"
    def __reduce__(self):
        return (self.__class__, self._iters)
    def __reduce_ex__(self, _):
        return self.__reduce__()
    def __copy__(self):
        return self.__class__(*_coconut.map(_coconut.copy.copy, self._iters))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class enumerate(_coconut.enumerate):
    __slots__ = ("_iter", "_start")
    if hasattr(_coconut.enumerate, "__doc__"):
        __doc__ = _coconut.enumerate.__doc__
    def __new__(cls, iterable, start=0):
        new_enumerate = _coconut.enumerate.__new__(cls, iterable, start)
        new_enumerate._iter, new_enumerate._start = iterable, start
        return new_enumerate
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(_coconut_igetitem(self._iter, index), self._start + (0 if index.start is None else index.start if index.start >= 0 else len(self._iter) + index.start))
        return (self._start + index, _coconut_igetitem(self._iter, index))
    def __len__(self):
        return _coconut.len(self._iter)
    def __repr__(self):
        return "enumerate(" + _coconut.repr(self._iter) + ", " + _coconut.repr(self._start) + ")"
    def __reduce__(self):
        return (self.__class__, (self._iter, self._start))
    def __reduce_ex__(self, _):
        return self.__reduce__()
    def __copy__(self):
        return self.__class__(_coconut.copy.copy(self._iter), self._start)
    def __fmap__(self, func):
        return _coconut_map(func, self)
class count(object):
    """count(start, step) returns an infinite iterator starting at start and increasing by step."""
    __slots__ = ("start", "step")
    def __init__(self, start=0, step=1):
        self.start, self.step = start, step
    def __iter__(self):
        while True:
            yield self.start
            self.start += self.step
    def __contains__(self, elem):
        return elem >= self.start and (elem - self.start) % self.step == 0
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice) and (index.start is None or index.start >= 0) and (index.stop is None or index.stop >= 0):
            if index.stop is None:
                return self.__class__(self.start + (index.start if index.start is not None else 0), self.step * (index.step if index.step is not None else 1))
            if _coconut.isinstance(self.start, _coconut.int) and _coconut.isinstance(self.step, _coconut.int):
                return _coconut.range(self.start + self.step * (index.start if index.start is not None else 0), self.start + self.step * index.stop, self.step * (index.step if index.step is not None else 1))
            return _coconut_map(self.__getitem__, _coconut.range(index.start if index.start is not None else 0, index.stop, index.step if index.step is not None else 1))
        if index >= 0:
            return self.start + self.step * index
        raise _coconut.IndexError("count indices must be positive")
    def count(self, elem):
        """Count the number of times elem appears in the count."""
        return int(elem in self)
    def index(self, elem):
        """Find the index of elem in the count."""
        if elem not in self:
            raise _coconut.ValueError(_coconut.repr(elem) + " is not in count")
        return (elem - self.start) // self.step
    def __repr__(self):
        return "count(" + _coconut.str(self.start) + ", " + _coconut.str(self.step) + ")"
    def __hash__(self):
        return _coconut.hash((self.start, self.step))
    def __reduce__(self):
        return (self.__class__, (self.start, self.step))
    def __copy__(self):
        return self.__class__(self.start, self.step)
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.start == other.start and self.step == other.step
    def __fmap__(self, func):
        return _coconut_map(func, self)
class groupsof(object):
    """groupsof(n, iterable) splits iterable into groups of size n.
    If the length of the iterable is not divisible by n, the last group may be of size < n."""
    __slots__ = ("group_size", "iter")
    def __init__(self, n, iterable):
        self.iter = iterable
        try:
            self.group_size = _coconut.int(n)
        except _coconut.ValueError:
            raise _coconut.TypeError("group size must be an int; not %r" % (n,))
        if self.group_size <= 0:
            raise _coconut.ValueError("group size must be > 0; not %s" % (self.group_size,))
    def __iter__(self):
        loop, iterator = True, _coconut.iter(self.iter)
        while loop:
            group = []
            for _ in _coconut.range(self.group_size):
                try:
                    group.append(_coconut.next(iterator))
                except _coconut.StopIteration:
                    loop = False
                    break
            if group:
                yield _coconut.tuple(group)
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "groupsof(%r)" % (_coconut.repr(self.iter),)
    def __reduce__(self):
        return (self.__class__, (self.group_size, self.iter))
    def __copy__(self):
        return self.__class__(self.group_size, _coconut.copy.copy(self.iter))
    def __fmap__(self, func):
        return _coconut_map(func, self)
def recursive_iterator(func):
    """Decorator that optimizes a function for iterator recursion."""
    tee_store, backup_tee_store = {}, []
    @_coconut.functools.wraps(func)
    def recursive_iterator_func(*args, **kwargs):
        key, use_backup = (args, _coconut.frozenset(kwargs)), False
        try:
            hash(key)
        except _coconut.Exception:
            try:
                key = _coconut.pickle.dumps(key, -1)
            except _coconut.Exception:
                use_backup = True
        if use_backup:
            for i, (k, v) in _coconut.enumerate(backup_tee_store):
                if k == key:
                    to_tee, store_pos = v, i
                    break
            else:  # no break
                to_tee, store_pos = func(*args, **kwargs), None
            to_store, to_return = _coconut_tee(to_tee)
            if store_pos is None:
                backup_tee_store.append([key, to_store])
            else:
                backup_tee_store[store_pos][1] = to_store
        else:
            tee_store[key], to_return = _coconut_tee(tee_store.get(key) or func(*args, **kwargs))
        return to_return
    return recursive_iterator_func
def addpattern(base_func):
    """Decorator to add a new case to a pattern-matching function,
    where the new case is checked last."""
    def pattern_adder(func):
        @_coconut_tco
        @_coconut.functools.wraps(func)
        def add_pattern_func(*args, **kwargs):
            try:
                return base_func(*args, **kwargs)
            except _coconut_MatchError:
                return _coconut_tail_call(func, *args, **kwargs)
        return add_pattern_func
    return pattern_adder
def prepattern(base_func):
    """DEPRECATED: Use addpattern instead."""
    def pattern_prepender(func):
        return addpattern(func)(base_func)
    return pattern_prepender
class _coconut_partial(object):
    __slots__ = ("func", "_argdict", "_arglen", "_stargs", "keywords")
    if hasattr(_coconut.functools.partial, "__doc__"):
        __doc__ = _coconut.functools.partial.__doc__
    def __init__(self, func, argdict, arglen, *args, **kwargs):
        self.func, self._argdict, self._arglen, self._stargs, self.keywords = func, argdict, arglen, args, kwargs
    def __reduce__(self):
        return (self.__class__, (self.func, self._argdict, self._arglen) + self._stargs, self.keywords)
    def __setstate__(self, keywords):
        self.keywords = keywords
    @property
    def args(self):
        return _coconut.tuple(self._argdict.get(i) for i in _coconut.range(self._arglen)) + self._stargs
    def __call__(self, *args, **kwargs):
        callargs = []
        argind = 0
        for i in _coconut.range(self._arglen):
            if i in self._argdict:
                callargs.append(self._argdict[i])
            elif argind >= _coconut.len(args):
                raise _coconut.TypeError("expected at least " + _coconut.str(self._arglen - _coconut.len(self._argdict)) + " argument(s) to " + _coconut.repr(self))
            else:
                callargs.append(args[argind])
                argind += 1
        callargs += self._stargs
        callargs += args[argind:]
        kwargs.update(self.keywords)
        return self.func(*callargs, **kwargs)
    def __repr__(self):
        args = []
        for i in _coconut.range(self._arglen):
            if i in self._argdict:
                args.append(_coconut.repr(self._argdict[i]))
            else:
                args.append("?")
        for arg in self._stargs:
            args.append(_coconut.repr(arg))
        return _coconut.repr(self.func) + "$(" + ", ".join(args) + ")"
def makedata(data_type, *args, **kwargs):
    """Call the original constructor of the given data type or class with the given arguments."""
    if _coconut.hasattr(data_type, "_make") and (_coconut.issubclass(data_type, _coconut.tuple) or _coconut.isinstance(data_type, _coconut.tuple)):
        return data_type._make(args, **kwargs)
    return _coconut.super(data_type, data_type).__new__(data_type, *args, **kwargs)
def datamaker(data_type):
    """DEPRECATED: Use makedata instead."""
    return _coconut.functools.partial(makedata, data_type)
def consume(iterable, keep_last=0):
    """consume(iterable, keep_last) fully exhausts iterable and return the last keep_last elements."""
    return _coconut.collections.deque(iterable, maxlen=keep_last)  # fastest way to exhaust an iterator
class starmap(_coconut.itertools.starmap):
    __slots__ = ("_func", "_iter")
    if hasattr(_coconut.itertools.starmap, "__doc__"):
        __doc__ = _coconut.itertools.starmap.__doc__
    def __new__(cls, function, iterable):
        new_map = _coconut.itertools.starmap.__new__(cls, function, iterable)
        new_map._func, new_map._iter = function, iterable
        return new_map
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(self._func, _coconut_igetitem(self._iter, index))
        return self._func(*_coconut_igetitem(self._iter, index))
    def __reversed__(self):
        return self.__class__(self._func, *_coconut_reversed(self._iter))
    def __len__(self):
        return _coconut.len(self._iter)
    def __repr__(self):
        return "starmap(" + _coconut.repr(self._func) + ", " + _coconut.repr(self._iter) + ")"
    def __reduce__(self):
        return (self.__class__, (self._func, self._iter))
    def __reduce_ex__(self, _):
        return self.__reduce__()
    def __copy__(self):
        return self.__class__(self._func, _coconut.copy.copy(self._iter))
    def __fmap__(self, func):
        return self.__class__(_coconut_forward_compose(self._func, func), self._iter)
def fmap(func, obj):
    """fmap(func, obj) creates a copy of obj with func applied to its contents.
    Override by defining .__fmap__(func)."""
    if _coconut.hasattr(obj, "__fmap__"):
        return obj.__fmap__(func)
    args = _coconut_starmap(func, obj.items()) if _coconut.isinstance(obj, _coconut.abc.Mapping) else _coconut_map(func, obj)
    if _coconut.isinstance(obj, _coconut.tuple) and _coconut.hasattr(obj, "_make"):
        return obj._make(args)
    if _coconut.isinstance(obj, (_coconut.map, _coconut.range, _coconut.abc.Iterator)):
        return args
    if _coconut.isinstance(obj, _coconut.str):
        return "".join(args)
    return obj.__class__(args)
_coconut_MatchError, _coconut_count, _coconut_enumerate, _coconut_reversed, _coconut_map, _coconut_starmap, _coconut_tee, _coconut_zip, reduce, takewhile, dropwhile = MatchError, count, enumerate, reversed, map, starmap, tee, zip, _coconut.functools.reduce, _coconut.itertools.takewhile, _coconut.itertools.dropwhile

# Compiled Coconut: -----------------------------------------------------------



PUZZLE = '29917128875332952564321392569634257121244516819997569284938677239676779378822158323549832814412597817651244117851771257438674567254146559419528411463781241159837576747416543451994579655175322397355255587935456185669334559882554936642122347526466965746273596321419312386992922582836979771421518356285534285825212798113159911272923448284681544657616654285632235958355867722479252256292311384799669645293812691169936746744856227797779513997329663235176153745581296191298956836998758194274865327383988992499115472925731787228592624911829221985925935268785757854569131538763133427434848767475989173579655375125972435359317237712667658828722623837448758528395981635746922144957695238318954845799697142491972626942976788997427135797297649149849739186827185775786254552866371729489943881272817466129271912247236569141713377483469323737384967871876982476485658337183881519295728697121462266226452265259877781881868585356333494916519693683238733823362353424927852348119426673294798416314637799636344448941782774113142925315947664869341363354235389597893211532745789957591898692253157726576488811769461354938575527273474399545366389515353657644736458182565245181653996192644851687269744491856672563885457872883368415631469696994757636288575816146927747179133188841148212825453859269643736199836818121559198563122442483528316837885842696283932779475955796132242682934853291737434482287486978566652161245555856779844813283979453489221189332412315117573259531352875384444264457373153263878999332444178577127433891164266387721116357278222665798584824336957648454426665495982221179382794158366894875864761266695773155813823291684611617853255857774422185987921219618596814446229556938354417164971795294741898631698578989231245376826359179266783767935932788845143542293569863998773276365886375624694329228686284863341465994571635379257258559894197638117333711626435669415976255967412994139131385751822134927578932521461677534945328228131973291962134523589491173343648964449149716696761218423314765168285342711137126239639867897341514131244859826663281981251614843274762372382114258543828157464392'


'''
In [67]: reduce( (a,x) -> (x |> int) + a, '101', 0)
Out[67]: 2
'''

def lshift(seq, n=1):
    return seq[n:] + seq[:n]

def captcha(*_coconut_match_to_args, **_coconut_match_to_kwargs):
    _coconut_match_check = False
    if (_coconut.len(_coconut_match_to_args) == 2) and ("acc" not in _coconut_match_to_kwargs) and (_coconut.isinstance(_coconut_match_to_args[1], _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to_args[1]) == 2):
        _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("acc")
        x = _coconut_match_to_args[1][0]
        y = _coconut_match_to_args[1][1]
        if not _coconut_match_to_kwargs:
            acc = _coconut_match_temp_0
            _coconut_match_check = True
    if not _coconut_match_check:
        _coconut_match_err = _coconut_MatchError("pattern-matching failed for " "'def captcha(acc, (x,y)):'" " in " + _coconut.repr(_coconut.repr(_coconut_match_to_args)))
        _coconut_match_err.pattern = 'def captcha(acc, (x,y)):'
        _coconut_match_err.value = _coconut_match_to_args
        raise _coconut_match_err

    x = (int)(x)
    y = (int)(y)
    if x == y:
        return acc + y
    else:
        return acc

@_coconut_tco
def part1(seq):
    return _coconut_tail_call((_coconut_partial(reduce, {0: captcha, 2: 0}, 3)), zip(seq, (lshift)(seq)))

@_coconut_tco
def part2(seq):
    halflen = len(seq) // 2
    return _coconut_tail_call((_coconut_partial(reduce, {0: captcha, 2: 0}, 3)), zip(seq, (_coconut_partial(lshift, {1: halflen}, 2))(seq)))


(print)('Part1')
(print)((part1)(PUZZLE))

##### -- PART2 

(print)('Part2')
(print)((part2)(PUZZLE))
