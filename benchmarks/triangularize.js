var bench = require('microbench'),
    vander = require('ndarray-vandermonde'),
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    fill = require('ndarray-fill'),
    qr = require('../lib'),
    show = require('ndarray-show')

bench.usePlugin('clitable');

bench.category('QR factorization',function() {

  var m = 10001,
      n = 10 

  bench.fixtures('d', pool.zeros([m],'float64'))
  bench.fixtures('A', pool.zeros([m,n],'float64'))
  bench.fixtures('triangularize',qr.triangularize)

  bench.perTestFixtures('matrix setup',function(fixtures) {
    fill(fixtures.A,function(i,j) {
      return Math.pow(i/(m-1),j)
    })
  })

  bench.test('v3',function(fixtures) {
    fixtures.triangularize(fixtures.A, fixtures.d)
  })

}).run()
