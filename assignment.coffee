qm = require('qminer')
loader = require('qminer-data-loader')


base = new qm.Base(
    mode: 'createClean',
    schema: [
      name: 'tweets',
      fields: [
        { name: 'text', type: 'string' },
        { name: 'target', type: 'float' }
        ]
    ]
)
