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
while (match = extraction_regex.exec(data))
  result.push {id: parseInt(match[1]), categories: create_categories(match[2]), text: match[match.length - 1]}

fs.unlinkSync('json_lines_dataset.txt')
for i in result
  fs.appendFileSync 'json_lines_dataset.txt', JSON.stringify(i) + "\n"

n_articles = base.store("articles").loadJson "json_lines_dataset.txt"

# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'tfidf', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'en', # none, en, [...]
#         stemmer: 'porter' # porter, none
#     },
#     normalize: true
# }



# feature_space = new qm.FeatureSpace(base, feature_space_args)
# feature_space.updateRecords base.store('articles').allRecords

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
#   [ 'GREL', 20 ], used RELIGION
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
#   [ 'GJOB', 158 ], used
#   [ 'GSPO', 178 ],
#   [ 'CCAT', 205 ],
#   [ 'GCRIM', 260 ],
#   [ 'ECAT', 297 ],
#   [ 'GVIO', 338 ],
#   [ 'GDIP', 387 ],
#   [ 'GPOL', 477 ], used DOMESTIC POLITICS
#   [ 'GCAT', 1955 ] ]

# training and validation set

# validationSetSize ∈ (0, 1) is the appoximate part of the recordSet
# to use for validation
# returns a 2 element array. the training set is first, the validation
# set second
makeVTSets = (recordSet, validationSetSize) ->
  total = recordSet.length
  n = Math.round(total * validationSetSize)
  vs = recordSet.sample n
  ts = recordSet.setDiff vs
  return [ts, vs]



# cross validation
#

makeTrainTestSets = (trainTest, folds) ->
  l = trainTest.length
  testSize = Math.round(l / folds)
  start = 0
  end = start + testSize
  trainTestIds = (r.$id for r in trainTest)
  console.log trainTest
  result = []
  while end <= l
    tt = trainTest.clone()
    tr = tt.filter (rec) -> (rec.$id in trainTestIds[start...end])
    ts = trainTest.clone()
    test = ts.setDiff tr
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

trainModel = (base, trainingSet, modelParams, fsArgs, category) ->
  features = new qm.FeatureSpace base, fsArgs
  features.updateRecords trainingSet
  SVC = new qm.analytics.SVC(modelParams)
  SVC.fit(features.extractSparseMatrix(trainingSet),
          makeTarget(trainingSet, category))
  return [SVC, features]

evaluateSVC = (SVC, featureSpace, testSet, category) ->
  predictions = SVC.predict featureSpace.extractSparseMatrix(testSet)
  trueTarget = makeTarget testSet, category
  results = {tp: 0, fp: 0, tn: 0, fn: 0}
  for i in [0...trueTarget.length]
    if (predictions[i] == trueTarget[i])
      switch predictions[i]
        when 1 then ++results['tp']
        when -1 then ++results['tn']
    else
      switch predictions[i]
        when 1 then ++results['fp']
        when -1 then ++results['fn']
  return results


crossValidation = (base, trainTestSet, folds, category, modelParams, fsArgs) ->
  trainTestSets = makeTrainTestSets trainTestSet, folds
  results = {tp: 0, fp: 0, tn: 0, fn: 0}
  for [ts, vs] in trainTestSets
    [SVC, features] = trainModel base, ts, modelParams, fsArgs, category
    #console.log (w for w in SVC.weights)
    results = evaluateSVC SVC, features, vs, category
  return results


precisionAndRecall = (resultsMap) ->
  precision = resultsMap['tp']/ (resultsMap['tp'] + resultsMap['fp'])
  recall = resultsMap['tp'] / (resultsMap['tp'] + resultsMap['fn'])
  return { precision: precision, recall: recall}

trainModelAllData = (base, category, modelParams, featureSpaceParams) ->
  trainModel base, base.store('articles').allRecords, modelParams, featureSpaceParams

tryModelParams = (records, category, paramsToTry, folds) ->
  results = []
  for [features, modelParams] in paramsToTry
    rdesc = {features: features, modelParams: modelParams}
    cvResults = crossValidation base, records, folds, category, modelParams, features
    rdesc['cvResults'] = cvResults
    pr = precisionAndRecall cvResults
    rdesc['precision'] = pr['precision']
    rdesc['recall'] = pr['recall']
    results.push rdesc
  results.sort (a, b) -> b.recall - a.recall #sort in descending order
  return {category: category, folds: folds, results: results}

validateResults = (base, trainingSet, validationSet, modelParams, features, category) ->
  [SVC, features] = trainModel base, trainingSet, modelParams, features, category
  return evaluateSVC SVC, features, validationSet, category


featureSpaceParams = [
  {
    type: 'text', source: 'articles', field: 'text',
    weight: 'none', # none, tf, idf, tfidf
    tokenizer: {
      type: 'simple',
      stopwords: 'none', # none, en, [...]
      stemmer: 'none' # porter, none
    },
    normalize: true
  },
  {
    type: 'text', source: 'articles', field: 'text',
    weight: 'tfidf', # none, tf, idf, tfidf
    tokenizer: {
      type: 'simple',
      stopwords: 'en', # none, en, [...]
      stemmer: 'porter' # porter, none
    },
    normalize: true
  },
  {
    type: 'text', source: 'articles', field: 'text',
    weight: 'tfidf', # none, tf, idf, tfidf
    tokenizer: {
      type: 'simple',
      stopwords: 'en', # none, en, [...]
      stemmer: 'porter' # porter, none
    },
    ngrams: 2
    normalize: true
  },
  {
    type: 'text', source: 'articles', field: 'text',
    weight: 'idf', # none, tf, idf, tfidf
    tokenizer: {
      type: 'simple',
      stopwords: 'en', # none, en, [...]
      stemmer: 'porter' # porter, none
    },
    normalize: true
  },
  {
    type: 'text', source: 'articles', field: 'text',
    weight: 'tfidf', # none, tf, idf, tfidf
    tokenizer: {
      type: 'simple',
      stopwords: 'en', # none, en, [...]
      stemmer: 'none' # porter, none
    },
    normalize: true
  },
  {
    type: 'text', source: 'articles', field: 'text',
    weight: 'tfidf', # none, tf, idf, tfidf
    tokenizer: {
      type: 'simple',
      stopwords: 'none', # none, en, [...]
      stemmer: 'porter' # porter, none
    },
    normalize: true
  },
  {
    type: 'text', source: 'articles', field: 'text',
    weight: 'tf', # none, tf, idf, tfidf
    tokenizer: {
      type: 'simple',
      stopwords: 'en', # none, en, [...]
      stemmer: 'porter' # porter, none
    },
    normalize: true
  }]

svcParams = (cmin, cmax, jmin, jmax, cjby) ->
  result = []
  for c in [Math.round(cmin/cjby)..Math.round(cmax/cjby)]
    for j in [Math.round(jmin/cjby)..Math.round(jmax/cjby)]
      result.push {maxTime: 30, c: c*cjby, j: j*cjby, verbose: false}
  result

#results as returned by tryModelParams
writeResults = (results, now) ->
  fname = 'cv-results-' + results.category + '-' + now + '.json'
  fs.writeFileSync fname, JSON.stringify(results, null, 2)

mergeParams = (cmin, cmax, jmin, jmax, cjby, fsArgs) ->
  r = []
  for f in fsArgs
    for cj in svcParams cmin, cmax, jmin, jmax, cjby
      r.push [f, cj]
  return r

writeValidationSet = (vs, now, category) ->
  fname = 'validation-set-' + category + '-' + now + '.json'
  fs.writeFileSync fname, vs.toJSON()

writeValidationResults = (vr, now, cat) ->
  fname = 'validation-results-' + cat + '-' + now +'.json'
  fs.writeFileSync fname JSON.stringify(vr, null, 2)

runParameterOpt = (category, validationSetSize) ->
  [ts, vs] = makeVTSets base.store('articles').allRecords, validationSetSize
  now = new Date()
  now = now.toISOString()
  writeValidationSet vs, now, category
  params = mergeParams(0.8, 1.4, 0.9, 1.5, 0.1, featureSpaceParams)
  results = tryModelParams ts, category, params, 10
  writeResults results, now
  best = results.results[0]
  bestModel = best.modelParams
  bestFeatures = best.features
  validationResults = validateResults base, ts, vs, bestModel, bestFeatures, category
  pr = precisionAndRecall validationRecall
  writeValidationResults [validationResults, pr, bestModel, bestFeatures], now, category
  return [validationResults, pr, bestModel, bestFeatures]

# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'none', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'none', # none, en, [...]
#         stemmer: 'none' # porter, none
#     },
#     normalize: true
# }
#
# category: 'GPOL'
# { precision: 0.8790560471976401, recall: 0.6247379454926625 }
# { tp: 298, fp: 41, tn: 1482, fn: 179 }
#
# ==================================================
# 
# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'tfidf', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'en', # none, en, [...]
#         stemmer: 'porter' # porter, none
#     },
#     normalize: true
# }
#
#
# category: 'GPOL'
# { tp: 335, fp: 54, tn: 1469, fn: 142 }
# { precision: 0.8611825192802056, recall: 0.7023060796645703 }
#
# ==================================================
#
# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'tfidf', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'en', # none, en, [...]
#         stemmer: 'porter' # porter, none
#     },
#     ngrams: 2
#     normalize: true
# }
#
# category: 'GPOL'
# { tp: 304, fp: 36, tn: 1487, fn: 173 }
# { precision: 0.8941176470588236, recall: 0.6373165618448637 }
#
# ==================================================
#
# 
# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'tfidf', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'en', # none, en, [...]
#         stemmer: 'porter' # porter, none
#     },
#     normalize: true
# }
#
# category: 'GREL'
# { tp: 10, fp: 1, tn: 1979, fn: 10 }
# { precision: 0.9090909090909091, recall: 0.5 }
#
# ==================================================
#
# 
# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'none', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'none', # none, en, [...]
#         stemmer: 'none' # porter, none
#     },
#     normalize: true
# }
#
# category: 'GREL'
# { tp: 0, fp: 0, tn: 1980, fn: 20 }
# { precision: NaN, recall: 0 }
#
# ==================================================
#
#
# 
# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'tf', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'en', # none, en, [...]
#         stemmer: 'porter' # porter, none
#     },
#     normalize: true
# }
#
# category: 'GREL'
# { tp: 4, fp: 1, tn: 1979, fn: 16 }
# { precision: 0.8, recall: 0.2 }
#
# ==================================================
#
# 
# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'tf', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'en', # none, en, [...]
#         stemmer: 'porter' # porter, none
#     },
#     normalize: true
# }
#
# category: GJOB
# { tp: 104, fp: 3, tn: 1839, fn: 54 }
# { precision: 0.9719626168224299, recall: 0.6582278481012658 }
#
# ==================================================
#
# 
# feature_space_args = {
#     type: 'text', source: 'articles', field: 'text',
#     weight: 'tfidf', # none, tf, idf, tfidf
#     tokenizer: {
#         type: 'simple',
#         stopwords: 'en', # none, en, [...]
#         stemmer: 'porter' # porter, none
#     },
#     normalize: true
# }
#
# category: GJOB
# { tp: 104, fp: 5, tn: 1837, fn: 54 }
# { precision: 0.9541284403669725, recall: 0.6582278481012658 }
