// This file is part of Parsito <http://github.com/ufal/parsito/>.
//
// Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
// Mathematics and Physics, Charles University in Prague, Czech Republic.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <limits>

#include "common.h"
#include "embedding.h"
#include "unilib/unicode.h"
#include "unilib/utf8.h"

namespace ufal {
namespace udpipe {
namespace parsito {

int embedding::lookup_word(const string& word, string& buffer) const {
  using namespace unilib;

  if (subform) {
    auto it = decomposed_forms.find(word);

    if (it != decomposed_forms.end() && it->second < 0) return it->second;

    if (it == decomposed_forms.end()) {
      string word_bow_eow;
      word_bow_eow.append("<").append(word).append(">");

      subforms.emplace_back();
      for (string_piece form(word_bow_eow); form.len; utf8::decode(form.str, form.len)) {
        string_piece substr = form;
        for (unsigned len = 0; len < 4 && substr.len; len++) {
          utf8::decode(substr.str, substr.len);
          if (len) {
            buffer.assign(form.str, substr.str - form.str);
            auto sub_it = dictionary.find(buffer);
            if (sub_it != dictionary.end())
              subforms.back().push_back(sub_it->second);
          }
        }
      }

      if (subforms.back().empty()) {
        if (unknown_index < 0) {
          subforms.pop_back();
          decomposed_forms.emplace(word, unknown_index);
          return unknown_index;
        }
        subforms.back().push_back(unknown_index);
      }

      sort(subforms.back().begin(), subforms.back().end());
      subforms.back().erase(unique(subforms.back().begin(), subforms.back().end()), subforms.back().end());

      unsigned id = subforms.size() - 1;
      assert((weights.size() / dimension) == id);
      assert(previous_weights.size() == id);

      previous_weights.emplace_back(dimension + 1, 0.f);
      weights.resize(weights.size() + dimension);
      it = decomposed_forms.emplace(word, id).first;
    }

    return it->second;
  }

  auto it = dictionary.find(word);
  if (it != dictionary.end()) return it->second;

  // We now apply several heuristics to find a match

  // Try locating uppercase/titlecase characters which we could lowercase
  bool first = true;
  unicode::category_t first_category = 0, other_categories = 0;
  for (auto&& chr : utf8::decoder(word)) {
    (first ? first_category : other_categories) |= unicode::category(chr);
    first = false;
  }

  if ((first_category & unicode::Lut) && (other_categories & unicode::Lut)) {
    // Lowercase all characters but the first
    buffer.clear();
    first = true;
    for (auto&& chr : utf8::decoder(word)) {
      utf8::append(buffer, first ? chr : unicode::lowercase(chr));
      first = false;
    }

    it = dictionary.find(buffer);
    if (it != dictionary.end()) return it->second;
  }

  if ((first_category & unicode::Lut) || (other_categories & unicode::Lut)) {
    utf8::map(unicode::lowercase, word, buffer);

    it = dictionary.find(buffer);
    if (it != dictionary.end()) return it->second;
  }

  // If the word starts with digit and contain only digits and non-letter characters
  // i.e. large number, date, time, try replacing it with first digit only.
  if ((first_category & unicode::N) && !(other_categories & unicode::L)) {
    buffer.clear();
    utf8::append(buffer, utf8::first(word));

    it = dictionary.find(buffer);
    if (it != dictionary.end()) return it->second;
  }

  return unknown_index;
}

int embedding::unknown_word() const {
  return unknown_index;
}

float* embedding::weight(int id) {
  return (float*) ((const embedding*)this)->weight(id);
}

const float* embedding::weight(int id) const {
  if (id < 0 || id * dimension >= weights.size()) return nullptr;

  if (subform) {
    if (!previous_weights[id].empty() && !previous_weights[id][dimension]) {
      fill_n(weights.begin() + id * dimension, dimension, 0.f);
      for (auto&& subform : subforms[id])
        for (unsigned j = 0; j < dimension; j++)
          weights[id * dimension + j] += weights[subform * dimension + j];

      float normalize = 1.f / float(subforms[id].size());
      for (unsigned j = 0; j < dimension; j++)
        weights[id * dimension + j] *= normalize;

      copy_n(weights.begin() + id * dimension, dimension, previous_weights[id].begin());
      previous_weights[id][dimension] = 1.f;
      active_subforms.push_back(id);
    }
  }

  return weights.data() + id * dimension;
}

void embedding::update_weights() {
  if (subform) {
    for (auto&& id : active_subforms) {
      float normalize = 1.f / float(subforms[id].size());
      for (unsigned j = 0; j < dimension; j++)
        weights[id * dimension + j] = (weights[id * dimension + j] - previous_weights[id][j]) * normalize;

      for (auto&& subform : subforms[id])
        for (unsigned j = 0; j < dimension; j++)
          weights[subform * dimension + j] += weights[id * dimension + j];

      previous_weights[id][dimension] = 0.f;
    }
    active_subforms.clear();
  }
}

void embedding::load(binary_decoder& data) {
  // Load dimemsion
  dimension = data.next_4B();

  updatable_index = numeric_limits<decltype(updatable_index)>::max();

  // Load dictionary
  dictionary.clear();
  string word;
  for (unsigned size = data.next_4B(); size; size--) {
    data.next_str(word);
    dictionary.emplace(word, dictionary.size());
  }

  unknown_index = data.next_1B() ? dictionary.size() : -1;
  subform = data.next_1B();

  // Load weights
  const float* weights_ptr = data.next<float>(dimension * (dictionary.size() + (unknown_index >= 0)));
  weights.assign(weights_ptr, weights_ptr + dimension * (dictionary.size() + (unknown_index >= 0)));

  subforms.resize(weights.size() / dimension);
  previous_weights.resize(weights.size() / dimension);
}

} // namespace parsito
} // namespace udpipe
} // namespace ufal
