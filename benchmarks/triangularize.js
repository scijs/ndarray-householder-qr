var bench = require('microbench'),
    vander = require('ndarray-vandermonde'),
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    fill = require('ndarray-fill'),
    qr = require('../lib'),
    qrPreUnroll = require('../historical/pre-unroll'),
    show = require('ndarray-show')

bench.usePlugin('clitable');

bench.category('QR factorization',function() {

  var m = 10001,
      n = 10 

  bench.fixtures('d', pool.zeros([m],'float64'))
  bench.fixtures('A', pool.zeros([m,n],'float64'))
  bench.fixtures('triv3',qr.triangularize)
  bench.fixtures('triv2',qrPreUnroll.triangularize)

  bench.perTestFixtures('matrix setup',function(fixtures) {
    fill(fixtures.A,function(i,j) {
      return Math.pow(i/(m-1),j)
    })
  })

  bench.test('v2',function(fixtures) {
    fixtures.triv2(fixtures.A, fixtures.d)
  })

  bench.test('v3',function(fixtures) {
    fixtures.triv3(fixtures.A, fixtures.d)
  })

}).run()
