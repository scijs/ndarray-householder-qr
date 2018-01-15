'use strict';

var blas = require('ndarray-blas-level1');
var abs = Math.abs;
var sqrt = Math.sqrt;
var dot = blas.dot;
var axpy = blas.axpy;
var nrm2 = blas.nrm2;
var scal = blas.scal;

function factor (A, d) {
  var j, m, n, i, s, ajj, Ajmj, dj, Ajmi;

  if (A.data.get) {
    throw new TypeError('factor():: support for generic ndarrays is not currently implemented');
  }

  if (A.dimension !== 2) {
    throw new TypeError('factor():: dimension of input matrix must be 2.');
  }

  m = A.shape[0];
  n = A.shape[1];

  if (m < n) {
    throw new TypeError('factor():: In input matrix A, number of rows m must be greater than number of column n.');
  }

  Ajmj = A.pick(null, 0);
  Ajmi = A.pick(null, 0);
  var AjmiOff = Ajmi.offset;
  var AjmjInc = A.stride[0] + A.stride[1];
  var AjmiInc1 = A.stride[0];
  var AjmiInc2 = A.stride[1];
  var AjmjShape = Ajmj.shape;
  var dInc = d.stride[0];
  var dOff = d.offset;

  for (j = 0;
    j < n;
    j++, AjmjShape[0]--, Ajmj.offset += AjmjInc, AjmiOff += AjmiInc1, dOff += dInc
  ) {
    s = nrm2(Ajmj);
    ajj = Ajmj.data[Ajmj.offset];
    dj = ajj > 0 ? -s : s;
    d.data[dOff] = dj;
    s = sqrt(s * (s + abs(ajj)));
    if (s === 0) return false;
    Ajmj.data[Ajmj.offset] = ajj - dj;
    scal(1 / s, Ajmj);

    for (i = j + 1, Ajmi.offset = AjmiOff + AjmiInc2 * i;
      i < n;
      i++, Ajmi.offset += AjmiInc2
    ) {
      s = -dot(Ajmj, Ajmi);
      axpy(s, Ajmj, Ajmi);
    }
  }

  return true;
}
var multiplyByQ = function multiplyByQ (A, b) {
  var j, Ajmj, yk;
  var n = A.shape[1];

  Ajmj = A.pick(null, n - 1).lo(n - 1);
  yk = b.lo(n - 1);

  var AjmjOffInc = A.stride[0] + A.stride[1];
  var AjmjShape = Ajmj.shape;
  var ykInc = yk.stride[0];
  var ykShape = yk.shape;

  for (j = n - 1;
    j >= 0;
    j--, Ajmj.offset -= AjmjOffInc, AjmjShape[0]++, yk.offset -= ykInc, ykShape[0]++
  ) {
    axpy(-dot(Ajmj, yk), Ajmj, yk);
  }

  return true;
};

var multiplyByQinv = function multiplyByQinv (A, b) {
  var j, Ajmj, yk;
  var n = A.shape[1];

  Ajmj = A.pick(null, 0).lo(0);
  yk = b.lo(0);

  var AjmjOffInc = A.stride[0] + A.stride[1];
  var AjmjShape = Ajmj.shape;
  var ykInc = yk.stride[0];
  var ykShape = yk.shape;

  for (j = 0;
    j < n;
    j++, Ajmj.offset += AjmjOffInc, AjmjShape[0]--, yk.offset += ykInc, ykShape[0]--
  ) {
    axpy(-dot(Ajmj, yk), Ajmj, yk);
  }

  return true;
};

var constructQ = function (QR, Q) {
  var i, j, Qj, QjInc;
  var n = Q.shape[1];
  var m = Q.shape[0];

  var Qdata = Q.data;
  var Qptr = Q.offset;
  var QInc0 = Q.stride[0];
  var QInc1 = Q.stride[1];

  for (j = 0; j < n; j++, Qptr = Q.offset + j * QInc0) {
    for (i = 0; i < m; i++, Qptr += QInc1) {
      Qdata[Qptr] = i === j ? 1 : 0;
    }
  }

  Qj = Q.pick(null, 0);
  QjInc = Q.stride[1];

  for (j = 0; j < n; j++, Qj.offset += QjInc) {
    multiplyByQ(QR, Qj);
  }

  return true;
};

var solve = function (QR, d, x) {
  var i, n, j, QRi, QRiInc01, QRiShape;
  var xj, xjInc, xInc, xPtr, dPtr, dInc, xData, dData;
  var QRiPtr, xjPtr, QRiXj, QRiInc1;

  if (QR.dimension !== 2) {
    throw new TypeError('factor():: dimension of input matrix must be 2.');
  }

  n = QR.shape[1];

  multiplyByQinv(QR, x);

  QRi = QR.pick(n - 2, null).lo(n - 1);
  QRiInc01 = QR.stride[1] + QR.stride[0];
  QRiInc1 = QR.stride[1];
  QRiShape = QRi.shape[0];

  xj = x.lo(n - 1);
  xjInc = x.stride[0];

  xPtr = x.offset + x.stride[0] * (n - 1);
  xInc = x.stride[0];
  xData = x.data;

  dPtr = d.offset + d.stride[0] * (n - 1);
  dInc = d.stride[0];
  dData = d.data;

  xData[xPtr] = xData[xPtr] / dData[dPtr];

  for (j = n - 2, xPtr -= xInc, dPtr -= dInc;
    j >= 0;
    j--, QRi.offset -= QRiInc01, QRiShape++, xj.offset -= xjInc, xPtr -= xInc, dPtr -= dInc
  ) {
    for (i = 0, QRiXj = 0, QRiPtr = QRi.offset, xjPtr = xj.offset;
      i < QRiShape;
      i++, QRiPtr += QRiInc1, xjPtr += xInc
    ) {
      QRiXj += QRi.data[QRiPtr] * xData[xjPtr];
    }

    xData[xPtr] -= QRiXj;
    xData[xPtr] /= dData[dPtr];
  }

  return true;
};

exports.factor = factor;
exports.multiplyByQ = multiplyByQ;
exports.multiplyByQinv = multiplyByQinv;
exports.constructQ = constructQ;
exports.solve = solve;

// Deprecations:
exports.triangularize = function () {
  console.warn('Warning: ndarray-householder-qr::triangularize() has been deprecated and renamed factor().');
  return exports.factor.apply(this, Array.prototype.slice.call(arguments));
};
