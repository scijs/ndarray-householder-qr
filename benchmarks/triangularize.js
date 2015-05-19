var bench = require('microbench'),
    vander = require('ndarray-vandermonde'),
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    fill = require('ndarray-fill'),
    qrv3 = require('../lib'),
    qrv2 = require('../historical/v2'),
    qrv1 = require('../historical/v1'),
    show = require('ndarray-show')

bench.usePlugin('clitable');

bench.category('QR factorization',function() {

  var m = 150,
      n = 60

  bench.fixtures('d', pool.zeros([m],'float64'))
  bench.fixtures('A', pool.zeros([m,n],'float64'))
  bench.fixtures('x', pool.zeros([n],'float64'))
  bench.fixtures('b', pool.zeros([m],'float64'))
  bench.fixtures('work', pool.zeros([qrv1.workVectorLength(m,n)],'float64'))
  bench.fixtures('triv3',qrv3.triangularize)
  bench.fixtures('triv2',qrv2.triangularize)
  bench.fixtures('triv1',qrv1.triangularize)
  bench.fixtures('solvev3',qrv3.solve)
  bench.fixtures('solvev2',qrv2.solve)
  bench.fixtures('solvev1',qrv1.solve)

  bench.perTestFixtures('matrix setup',function(fixtures) {
    fill(fixtures.A,function(i,j) {
      return Math.pow(i/(m-1),j)
    })
    fill(fixtures.x,function(i) {
      return 1 + Math.sin(i/n)
    });
    fill(fixtures.b,function(i) {
      return 1 + Math.sin(i/n)
    });
  })

  bench.test('Naively implemented algorithm',function(fixtures) {
    fixtures.triv1(fixtures.A, fixtures.work)
    fixtures.solvev1(fixtures.A, fixtures.work, fixtures.b, fixtures.x)
  })

  bench.test('Streamlined algorithm',function(fixtures) {
    fixtures.triv2(fixtures.A, fixtures.d)
    fixtures.solvev2(fixtures.A, fixtures.d, fixtures.b)
  })

  bench.test('Streamlined + unrolled algorithm',function(fixtures) {
    fixtures.triv3(fixtures.A, fixtures.d)
    fixtures.solvev3(fixtures.A, fixtures.d, fixtures.b)
  })

}).run()
