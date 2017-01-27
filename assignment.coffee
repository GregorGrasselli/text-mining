qm = require('qminer')
loader = require('qminer-data-loader')
fs = require 'fs'

base = new qm.Base(
    mode: 'createClean',
    schema: [
      {name: 'articles',
      fields: [
        { name: 'text', type: 'string' },
        { name: 'id', type: 'float' },
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
  

# result = []
# while (match = extraction_regex.exec(data))
#   result.push {id: match[1], categories: create_categories(match[2]), text: match[match.length - 1]}

# for i in result
#   fs.appendFile 'json_lines_dataset.txt', JSON.stringify(i) + "\n", (error) -> console.log("couldn't write " + i)

data = base.store("articles").loadJson "json_lines_dataset.txt"

feature_space = new qm.FeatureSpace(base, {
    type: 'text', source: 'articles', field: 'text',
    weight: 'tfidf', # none, tf, idf, tfidf
    tokenizer: {
        type: 'simple',
        stopwords: 'none', # none, en, [...]
        stemmer: 'none' # porter, none
    },
    ngrams: 1,
    normalize: true
})

null
