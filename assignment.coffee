qm = require('qminer')
#loader = require('qminer-data-loader')
fs = require 'fs'

base = new qm.Base(
    mode: 'createClean',
    schema: [
      {name: 'articles',
       fields: [
        { name: 'text', type: 'string' },
        { name: 'id', type: 'int' },
        { name: 'index', primary: true, type: 'int' },
        { name: 'categories', type: 'string_v' }
        ]},
    ]
)


data = ->
  fs.readFileSync 'dataset.txt', 'utf8'

data = data();null
extraction_regex = /([0-9]+) ((![A-Z]+ )+) (.+)/g
#parsed = extraction_regex.exec(data);null

create_categories = (cats) ->
  (s.substr(1) for s in cats.split " " when s.length > 0)


result = []
index = 0
while (match = extraction_regex.exec(data))
  result.push {id: parseInt(match[1]), categories: create_categories(match[2]), text: match[match.length - 1], index: index}
  ++index

fs.unlinkSync('json_lines_dataset.txt')
for i in result
  fs.appendFileSync 'json_lines_dataset.txt', JSON.stringify(i) + "\n"

n_articles = base.store("articles").loadJson "json_lines_dataset.txt"

feature_space_args = {
    type: 'text', source: 'articles', field: 'text',
    weight: 'tfidf', # none, tf, idf, tfidf
    tokenizer: {
        type: 'simple',
        stopwords: 'en', # none, en, [...]
        stemmer: 'porter' # porter, none
    },
    ngrams: 2,
    normalize: true
}

feature_space = new qm.FeatureSpace(base, feature_space_args)

listAllCategories = (base) ->
  categories = {}
  base.store("articles").each((rec) ->
    for cat in rec.categories
      if !categories[cat]
        categories[cat] = 1
      else
        categories[cat] += 1)

  categories = ([k, v] for k, v of categories).sort (a, b) -> a[1] - b[1]

# listAllCategories base

# [ [ 'GFAS', 1 ],
#   [ 'GTOUR', 5 ],
#   [ 'GOBIT', 5 ],
#   [ 'GREL', 20 ],
#   [ 'GODD', 23 ],
#   [ 'GWELF', 27 ],
#   [ 'GSCI', 28 ],
#   [ 'GENT', 41 ],
#   [ 'GPRO', 44 ],
#   [ 'MCAT', 46 ],
#   [ 'GWEA', 46 ],
#   [ 'GENV', 55 ],
#   [ 'GHEA', 67 ],
#   [ 'GDIS', 85 ],
#   [ 'GDEF', 116 ],
#   [ 'GVOTE', 119 ],
#   [ 'GJOB', 158 ],
#   [ 'GSPO', 178 ],
#   [ 'CCAT', 205 ],
#   [ 'GCRIM', 260 ],
#   [ 'ECAT', 297 ],
#   [ 'GVIO', 338 ],
#   [ 'GDIP', 387 ],
#   [ 'GPOL', 477 ],
#   [ 'GCAT', 1955 ] ]

# cross validation
#


makeTrainTestSets = (base, folds) ->
  l = base.store("articles").allRecords.length
  testSize = Math.round(l / folds)
  start = 0
  end = start + testSize
  result = []
  while end <= l
    tr= base.store('articles').allRecords.filter (rec) -> rec.index < start or rec.index >= end
    test = base.store('articles').allRecords.filter (rec) -> rec.index >= start and rec.index < end
    start = end
    end += testSize
    result.push [tr, test]
  return result


makeTarget = (ts, cat) ->
  target = ts.map((rec) ->
    if cat in rec.categories
      1                         #true
    else
      -1)                        #false
  return qm.la.Vector(target)

crossValidation = (base, folds, category) ->
  trainTestSets = makeTrainTestSets base, folds
  results = {tp: 0, fp: 0, tn: 0, fn: 0}
  for [ts, vs] in trainTestSets
    SVC = new qm.analytics.SVC({maxTime: 30})
    SVC.fit(feature_space.extractSparseMatrix(ts),
            makeTarget(ts, category))
    console.log (w for w in SVC.weights)
    vs.each((rec) ->
      sparseVector = feature_space.extractSparseVector({ text: rec.text })
      y = SVC.predict(sparseVector)
      console.log (c for c in rec.categories), y
      if (category in rec.categories) and y == 1 #true positive
        ++results['tp']
        console.log 'tp', '\n'
      else if (category in rec.categories) and y == -1 #false negative
        ++results['fn']
        console.log 'fn', '\n'
      else if y == 1            #false positive
        ++results['fp']
        console.log 'fp', '\n'
      else                      #true negative
        ++results['tn']
        console.log 'tn', '\n')
  return results
        
