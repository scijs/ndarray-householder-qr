var bench = require('microbench');
var pool = require('ndarray-scratch');
var fill = require('ndarray-fill');
var qrv3 = require('../');
var qrv2 = require('../historical/v2');
var qrv1 = require('../historical/v1');

bench.usePlugin('clitable');

bench.set('throwaway iterations', 200);
bench.set('default iterations', 100);

bench.category('QR factorization', function () {
  var m = 120;
  var n = 12;

  bench.fixtures('d', pool.zeros([m], 'float64'));
  bench.fixtures('A', pool.zeros([m, n], 'float64'));
  bench.fixtures('x', pool.zeros([n], 'float64'));
  bench.fixtures('b', pool.zeros([m], 'float64'));
  bench.fixtures('work', pool.zeros([qrv1.workVectorLength(m, n)], 'float64'));

  bench.perTestFixtures('matrix setup', function (fixtures) {
    fill(fixtures.A, function (i, j) {
      return Math.pow(i / (m - 1), j);
    });
    fill(fixtures.x, function (i) {
      return 1 + Math.sin(i / n);
    });
    fill(fixtures.b, function (i) {
      return 1 + Math.sin(i / n);
    });
  });

  bench.test('Naively implemented algorithm', function (fixtures) {
    qrv1.triangularize(fixtures.A, fixtures.work);
    qrv1.solve(fixtures.A, fixtures.work, fixtures.b, fixtures.x);
  });

  bench.test('Streamlined algorithm', function (fixtures) {
    qrv2.triangularize(fixtures.A, fixtures.d);
    qrv2.solve(fixtures.A, fixtures.d, fixtures.b);
  });

  bench.test('Streamlined + unrolled algorithm', function (fixtures) {
    qrv3.factor(fixtures.A, fixtures.d);
    qrv3.solve(fixtures.A, fixtures.d, fixtures.b);
  });
}).run();
