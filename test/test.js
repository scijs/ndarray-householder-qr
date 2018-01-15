'use strict';

var test = require('tape');
var qr = require('../');
var ndarray = require('ndarray');
var ndtest = require('ndarray-tests');
var pool = require('ndarray-scratch');
var ops = require('ndarray-ops');

test('3 x 2 QR factorization', function (t) {
  function createData () {
    var m = 3;
    var n = 2;

    var Acontent = ndarray([1, 0, 1, 1, 1, 2], [3, 2]);
    var A = pool.zeros([50, 70, 80]).pick(null, null, 14).lo(7, 19).hi(20, 30).step(7, 15);
    ops.assign(A, Acontent);

    return {
      A: A,
      m: m,
      n: n,
      d: pool.zeros([n]),
      b: ndarray([1, 2, 3]),
      y: pool.zeros([n]),
      Q: pool.zeros([m, m]),
      Qreduced: pool.zeros([m, n])
    };
  }

  function free (data) {
    pool.free(data.A);
    pool.free(data.d);
    pool.free(data.y);
    pool.free(data.Q);
    pool.free(data.Qreduced);
  }

  t.test('qr factorization', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;

    qr.factor(A, d);

    var v1 = ndarray([1.256, 0.46, 0.46]);
    var v2 = ndarray([1.122, 0.861]);

    t.ok(ndtest.approximatelyEqual(v1, A.pick(null, 0), 1e-3), 'first column is correct');
    t.ok(ndtest.approximatelyEqual(v2, A.pick(null, 1).lo(1), 1e-3), 'second column is correct');

    var dExp = ndarray([-1.732, -1.414]);
    t.ok(ndtest.approximatelyEqual(dExp, d, 1e-3), 'diagonal is correct');
    t.ok(Math.abs(A.get(0, 1) - (-1.732)) < 1e-3, 'superdiagonal is correct');

    free(data);
    t.end();
  });

  t.test('implicit multiplication by inverse of Q', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var b = data.b;

    qr.factor(A, d);
    t.ok(qr.multiplyByQinv(A, b), 'returns true on success');

    var QinvbExpected = ndarray([-3.464, -1.414, 0]);
    t.ok(ndtest.approximatelyEqual(QinvbExpected, b, 1e-3), 'Qinv * b is correct');

    free(data);
    t.end();
  });

  t.test('multiplying by Q', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;

    var x = ndarray([-3.464, -1.414, 0]);
    var QbExpected = ndarray([1, 2, 3]);

    qr.factor(A, d);
    t.ok(qr.multiplyByQ(A, x), 'returns true on success');

    t.ok(ndtest.approximatelyEqual(QbExpected, x, 1e-3), 'Q * b is correct');

    free(data);
    t.end();
  });

  t.test('explicit construction of Q', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var Q = data.Q;

    qr.factor(A, d);
    t.ok(qr.constructQ(A, Q), 'returns true on success');

    t.ok(ndtest.matrixIsOrthogonal(Q, 1e-3), 'Q is orthogonal');

    free(data);
    t.end();
  });

  t.test('explicit construction of reduced Q', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var Qreduced = data.Qreduced;

    qr.factor(A, d);
    t.ok(qr.constructQ(A, Qreduced), 'returns true on success');

    t.ok(ndtest.matrixColsAreNormalized(Qreduced, 1e-3), 'columns of Q are normalized');
    t.ok(ndtest.matrixColsAreOrthogonal(Qreduced, 1e-3), 'columns of Q are orthogonal');

    free(data);
    t.end();
  });

  t.test('solve', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var n = data.n;
    var y0 = ndarray([1, 2, 3]);
    var xExpected = ndarray([1, 1]);

    qr.factor(A, d);
    qr.solve(A, d, y0);

    t.ok(ndtest.approximatelyEqual(xExpected, y0.hi(n), 1e-4), 'solution is correct');

    free(data);
    t.end();
  });

  t.test('reusing factorization for more than one call to solve', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var n = data.n;
    var y1 = ndarray([1, 2, 3]);
    var y2 = ndarray([1, 2, 3]);
    var xExpected = ndarray([1, 1]);

    qr.factor(A, d);
    t.ok(qr.solve(A, d, y1), 'returns true on success');
    t.ok(qr.solve(A, d, y2), 'returns true on success');

    t.ok(ndtest.approximatelyEqual(xExpected, y1.hi(n), 1e-4));
    t.ok(ndtest.approximatelyEqual(xExpected, y2.hi(n), 1e-4));

    free(data);
    t.end();
  });
});

test('6 x 4 QR factorization', function (t) {
  function createData () {
    var m = 6;
    var n = 4;

    var Acontent = ndarray([
      1, 0, 0, 0,
      1, 1, 1, 1,
      1, 2, 4, 8,
      1, 3, 9, 27,
      1, 4, 16, 64,
      1, 5, 25, 125
    ], [m, n]);
    var A = pool.zeros([60, 80, 90]).pick(null, null, 14).lo(15, 19).step(8, 10).hi(m, n);
    ops.assign(A, Acontent);

    var Qbase = pool.zeros([80, 40]);
    var Q = Qbase.lo(4, 17).step(4, 3).hi(m, m);
    var QreducedBase = pool.zeros([27, 12]);
    var Qreduced = QreducedBase.lo(2, 3).step(3, 2).hi(m, n);
    var d = pool.zeros([14, 3, 12]).pick(3, 2, null).lo(1).hi(n);
    var bContent = ndarray([1, 2, 3, 4, 5, 6]);
    var b = pool.zeros([17, 8, 12]).pick(3, null, 4).lo(2).hi(m);
    ops.assign(b, bContent);
    var y = pool.zeros([n]);

    return {
      m: m,
      n: n,
      A: A,
      Q: Q,
      Qreduced: Qreduced,
      d: d,
      b: b,
      y: y
    };
  }

  function free (data) {
    pool.free(data.d);
    pool.free(data.Q);
    pool.free(data.Qreduced);
    pool.free(data.b);
    pool.free(data.d);
    pool.free(data.y);
  }

  t.test('factor', function () {
    var data = createData();
    var A = data.A;

    t.ok(qr.factor(A, data.d), 'returns true on success');

    var v1 = ndarray([1.187, 0.344, 0.344, 0.344, 0.344, 0.344]);
    var v2 = ndarray([-6.124, -1.089, 0.049, 0.269, 0.488, 0.708]);
    var v3 = ndarray([-22.454, 20.916, -1.268, -0.521, -0.305, 0.168]);
    var v4 = ndarray([-91.856, 99.563, 45.826, -1.101, -0.214, 0.861]);

    t.ok(ndtest.approximatelyEqual(v1, A.pick(null, 0), 1e-3), 'first column of A is correct');
    t.ok(ndtest.approximatelyEqual(v2, A.pick(null, 1), 1e-3), 'second column of A is correct');
    t.ok(ndtest.approximatelyEqual(v3, A.pick(null, 2), 1e-3), 'third column of A is correct');
    t.ok(ndtest.approximatelyEqual(v4, A.pick(null, 3), 1e-3), 'fourth column of A is correct');

    free(data);
    t.end();
  });

  t.test('multiplying by inverse of Q', function (t) {
    var data = createData();
    var A = data.A;
    var b = data.b;
    var d = data.d;

    var QinvbExpected = ndarray([-8.573, 4.183, 0.000, 0.000, 0.000, 0.000]);
    qr.factor(A, d);
    t.ok(qr.multiplyByQinv(A, b), 'returns true on success');
    t.ok(ndtest.approximatelyEqual(QinvbExpected, b, 1e-3), 'Qinv * b is correct');

    free(data);
    t.end();
  });

  t.test('multiplying by Q', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;

    var x = ndarray([-8.573, 4.183, 0.000, 0.000, 0.000, 0.000]);
    var QbExpected = ndarray([1, 2, 3, 4, 5, 6]);

    qr.factor(A, d);
    t.ok(qr.multiplyByQ(A, x), 'returns true on success');

    t.ok(ndtest.approximatelyEqual(QbExpected, x, 1e-3), 'Q * b is correct');

    free(data);
    t.end();
  });

  t.test('constructing Q explicitly', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var Q = data.Q;

    qr.factor(A, d);
    t.ok(qr.constructQ(A, Q), 'returns true on success');

    t.ok(ndtest.matrixIsOrthogonal(Q, 1e-3), 'is orthogonal');

    free(data);
    t.end();
  });

  t.test('constructing reduced Q explicitly', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var Qreduced = data.Qreduced;

    qr.factor(A, d);
    t.ok(qr.constructQ(A, Qreduced), 'returns true on success');

    t.ok(ndtest.matrixColsAreNormalized(Qreduced, 1e-8), 'Columns of Q are normalized');
    t.ok(ndtest.matrixColsAreOrthogonal(Qreduced, 1e-8), 'Columns of Q are orthogonal');

    free(data);
    t.end();
  });

  t.test('solving', function (t) {
    var data = createData();
    var A = data.A;
    var d = data.d;
    var b = data.b;

    var xExpected = ndarray([1, 1, 0, 0, 0, 0]);

    qr.factor(A, d);
    t.ok(qr.solve(A, d, b), 'returns true on success');

    t.ok(ndtest.approximatelyEqual(xExpected, b, 1e-8), 'solution is correct');

    free(data);
    t.end();
  });
});
