// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// json_lite: a strict JSON reader over a caller-given token pool, and a writer into a
// caller-given buffer. Neither allocates, throws or uses RTTI, so both run unchanged on the
// device (ESP-IDF builds C++ without exceptions) and on the host.
//
// The reader validates the whole text against RFC 8259 (strings with their escapes and UTF-16
// surrogate pairs, the number grammar, no trailing text) and records each value as one token in
// document order; strings are unescaped only when read. Nesting is bounded by `kMaxDepth`, so the
// recursive descent has a fixed worst-case stack on the device.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mist_core {
namespace json {

enum class Type : uint8_t { Null, False, True, Number, String, Array, Object };

enum class ParseError : uint8_t { None, Syntax, Depth, Capacity, TooLarge };

constexpr uint32_t kMaxDepth = 32;
constexpr uint32_t kNoToken = 0xFFFFFFFFu;

// One value in document order. For a string, `start` and `end` bound the raw text between the
// quotes; for every other value, its text. `next` is the index of the token after this value's
// subtree, so a reader skips a whole subtree in one step. `count` is an array's element count or
// an object's member count; a member is its key's string token followed by its value's subtree.
struct Token {
  uint32_t start;
  uint32_t end;
  uint32_t next;
  uint32_t count;
  Type type;
};

class Doc;

namespace detail {

inline int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

inline bool read_hex4(const char* p, uint32_t* out) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    const int digit = hex_value(p[i]);
    if (digit < 0) return false;
    value = value * 16u + static_cast<uint32_t>(digit);
  }
  *out = value;
  return true;
}

// Decodes the raw text of a validated string token into UTF-8, one byte at a time.
class StringDecoder {
 public:
  StringDecoder(const char* raw, size_t len) : p_(raw), end_(raw + len) {}

  // Returns false at the end of the string.
  bool next(uint8_t* out) {
    if (pending_pos_ < pending_len_) {
      *out = pending_[pending_pos_++];
      return true;
    }
    if (p_ >= end_) return false;
    const char c = *p_++;
    if (c != '\\') {
      *out = static_cast<uint8_t>(c);
      return true;
    }
    const char e = *p_++;
    switch (e) {
      case '"': *out = '"'; return true;
      case '\\': *out = '\\'; return true;
      case '/': *out = '/'; return true;
      case 'b': *out = '\b'; return true;
      case 'f': *out = '\f'; return true;
      case 'n': *out = '\n'; return true;
      case 'r': *out = '\r'; return true;
      case 't': *out = '\t'; return true;
      default: break;
    }
    // `\uXXXX`, validated by the parser, a high surrogate always followed by its low half.
    uint32_t code = 0;
    read_hex4(p_, &code);
    p_ += 4;
    if (code >= 0xD800 && code <= 0xDBFF) {
      uint32_t low = 0;
      read_hex4(p_ + 2, &low);
      p_ += 6;
      code = 0x10000u + ((code - 0xD800u) << 10) + (low - 0xDC00u);
    }
    encode_utf8(code);
    *out = pending_[pending_pos_++];
    return true;
  }

 private:
  void encode_utf8(uint32_t code) {
    pending_pos_ = 0;
    if (code < 0x80) {
      pending_[0] = static_cast<uint8_t>(code);
      pending_len_ = 1;
    } else if (code < 0x800) {
      pending_[0] = static_cast<uint8_t>(0xC0 | (code >> 6));
      pending_[1] = static_cast<uint8_t>(0x80 | (code & 0x3F));
      pending_len_ = 2;
    } else if (code < 0x10000) {
      pending_[0] = static_cast<uint8_t>(0xE0 | (code >> 12));
      pending_[1] = static_cast<uint8_t>(0x80 | ((code >> 6) & 0x3F));
      pending_[2] = static_cast<uint8_t>(0x80 | (code & 0x3F));
      pending_len_ = 3;
    } else {
      pending_[0] = static_cast<uint8_t>(0xF0 | (code >> 18));
      pending_[1] = static_cast<uint8_t>(0x80 | ((code >> 12) & 0x3F));
      pending_[2] = static_cast<uint8_t>(0x80 | ((code >> 6) & 0x3F));
      pending_[3] = static_cast<uint8_t>(0x80 | (code & 0x3F));
      pending_len_ = 4;
    }
  }

  const char* p_;
  const char* end_;
  uint8_t pending_[4] = {0, 0, 0, 0};
  uint8_t pending_len_ = 0;
  uint8_t pending_pos_ = 0;
};

}  // namespace detail

// A view of one value in a parsed `Doc`; an absent value (a missing key, an index past the end)
// is not `ok()`, and every accessor on it fails.
class Value {
 public:
  Value() = default;
  Value(const Doc* doc, uint32_t index) : doc_(doc), index_(index) {}

  bool ok() const { return doc_ != nullptr && index_ != kNoToken; }
  uint32_t index() const { return index_; }
  inline Type type() const;
  bool is(Type wanted) const { return ok() && type() == wanted; }
  bool is_null() const { return is(Type::Null); }
  bool is_string() const { return is(Type::String); }
  bool is_number() const { return is(Type::Number); }
  bool is_array() const { return is(Type::Array); }
  bool is_object() const { return is(Type::Object); }
  bool is_bool() const { return is(Type::True) || is(Type::False); }

  // Number of array elements or object members; 0 for anything else.
  inline uint32_t size() const;
  // The `i`th array element, or the value of the `i`th object member.
  inline Value at(uint32_t i) const;
  // The key of the `i`th object member.
  inline Value key_at(uint32_t i) const;
  // The value of the first member named `key`.
  inline Value get(const char* key) const;

  inline bool get_bool(bool* out) const;
  // An integer without fraction or exponent that fits in 64 bits.
  inline bool get_int(int64_t* out) const;
  inline bool get_double(double* out) const;
  // Writes the unescaped string and a terminating NUL; fails when it does not fit in `cap`.
  inline bool get_string(char* out, size_t cap, size_t* len = nullptr) const;
  // Whether this is a string whose unescaped bytes equal `text`.
  inline bool equals(const char* text) const;

 private:
  inline const Token* token() const;
  inline const char* raw() const;

  const Doc* doc_ = nullptr;
  uint32_t index_ = kNoToken;
};

class Doc {
 public:
  Doc(Token* pool, uint32_t capacity) : pool_(pool), capacity_(capacity) {}

  // Parses `len` bytes of `text`, which must outlive the `Doc`. The previous parse is discarded.
  ParseError parse(const char* text, size_t len) {
    text_ = text;
    len_ = len;
    pos_ = 0;
    used_ = 0;
    if (len >= kNoToken) return fail(ParseError::TooLarge);
    skip_ws();
    const ParseError error = value(0);
    if (error != ParseError::None) return fail(error);
    skip_ws();
    if (pos_ != len_) return fail(ParseError::Syntax);
    return ParseError::None;
  }

  Value root() const { return used_ > 0 ? Value(this, 0) : Value(); }
  uint32_t used() const { return used_; }
  const Token& token(uint32_t i) const { return pool_[i]; }
  const char* text() const { return text_; }

 private:
  ParseError fail(ParseError error) {
    used_ = 0;
    return error;
  }

  void skip_ws() {
    while (pos_ < len_) {
      const char c = text_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      ++pos_;
    }
  }

  bool at_end() const { return pos_ >= len_; }
  char peek() const { return text_[pos_]; }

  bool new_token(Type type, uint32_t* index) {
    if (used_ >= capacity_) return false;
    *index = used_++;
    pool_[*index] = Token{static_cast<uint32_t>(pos_), static_cast<uint32_t>(pos_), *index + 1, 0,
                          type};
    return true;
  }

  bool literal(const char* word) {
    const size_t n = std::strlen(word);
    if (len_ - pos_ < n || std::memcmp(text_ + pos_, word, n) != 0) return false;
    pos_ += n;
    return true;
  }

  ParseError value(uint32_t depth) {
    if (at_end()) return ParseError::Syntax;
    const char c = peek();
    if (c == '{') return object(depth);
    if (c == '[') return array(depth);
    if (c == '"') return string();
    if (c == '-' || (c >= '0' && c <= '9')) return number();
    Type type;
    const char* word;
    if (c == 't') {
      type = Type::True;
      word = "true";
    } else if (c == 'f') {
      type = Type::False;
      word = "false";
    } else if (c == 'n') {
      type = Type::Null;
      word = "null";
    } else {
      return ParseError::Syntax;
    }
    uint32_t index = 0;
    if (!new_token(type, &index)) return ParseError::Capacity;
    if (!literal(word)) return ParseError::Syntax;
    pool_[index].end = static_cast<uint32_t>(pos_);
    return ParseError::None;
  }

  ParseError object(uint32_t depth) {
    if (depth + 1 > kMaxDepth) return ParseError::Depth;
    uint32_t index = 0;
    if (!new_token(Type::Object, &index)) return ParseError::Capacity;
    ++pos_;
    skip_ws();
    if (!at_end() && peek() == '}') {
      ++pos_;
      return close(index);
    }
    for (;;) {
      skip_ws();
      if (at_end() || peek() != '"') return ParseError::Syntax;
      ParseError error = string();
      if (error != ParseError::None) return error;
      skip_ws();
      if (at_end() || peek() != ':') return ParseError::Syntax;
      ++pos_;
      skip_ws();
      error = value(depth + 1);
      if (error != ParseError::None) return error;
      ++pool_[index].count;
      skip_ws();
      if (at_end()) return ParseError::Syntax;
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      if (peek() != '}') return ParseError::Syntax;
      ++pos_;
      return close(index);
    }
  }

  ParseError array(uint32_t depth) {
    if (depth + 1 > kMaxDepth) return ParseError::Depth;
    uint32_t index = 0;
    if (!new_token(Type::Array, &index)) return ParseError::Capacity;
    ++pos_;
    skip_ws();
    if (!at_end() && peek() == ']') {
      ++pos_;
      return close(index);
    }
    for (;;) {
      skip_ws();
      const ParseError error = value(depth + 1);
      if (error != ParseError::None) return error;
      ++pool_[index].count;
      skip_ws();
      if (at_end()) return ParseError::Syntax;
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      if (peek() != ']') return ParseError::Syntax;
      ++pos_;
      return close(index);
    }
  }

  ParseError close(uint32_t index) {
    pool_[index].end = static_cast<uint32_t>(pos_);
    pool_[index].next = used_;
    return ParseError::None;
  }

  ParseError string() {
    uint32_t index = 0;
    if (!new_token(Type::String, &index)) return ParseError::Capacity;
    ++pos_;
    pool_[index].start = static_cast<uint32_t>(pos_);
    while (!at_end()) {
      const unsigned char c = static_cast<unsigned char>(peek());
      if (c == '"') {
        pool_[index].end = static_cast<uint32_t>(pos_);
        ++pos_;
        return ParseError::None;
      }
      if (c < 0x20) return ParseError::Syntax;
      if (c != '\\') {
        ++pos_;
        continue;
      }
      ++pos_;
      if (at_end()) return ParseError::Syntax;
      const char e = peek();
      ++pos_;
      if (e == '"' || e == '\\' || e == '/' || e == 'b' || e == 'f' || e == 'n' || e == 'r' ||
          e == 't') {
        continue;
      }
      if (e != 'u') return ParseError::Syntax;
      uint32_t code = 0;
      if (len_ - pos_ < 4 || !detail::read_hex4(text_ + pos_, &code)) return ParseError::Syntax;
      pos_ += 4;
      if (code >= 0xDC00 && code <= 0xDFFF) return ParseError::Syntax;
      if (code >= 0xD800 && code <= 0xDBFF) {
        uint32_t low = 0;
        if (len_ - pos_ < 6 || text_[pos_] != '\\' || text_[pos_ + 1] != 'u' ||
            !detail::read_hex4(text_ + pos_ + 2, &low) || low < 0xDC00 || low > 0xDFFF) {
          return ParseError::Syntax;
        }
        pos_ += 6;
      }
    }
    return ParseError::Syntax;
  }

  bool digits() {
    const size_t begin = pos_;
    while (!at_end() && peek() >= '0' && peek() <= '9') ++pos_;
    return pos_ > begin;
  }

  ParseError number() {
    uint32_t index = 0;
    if (!new_token(Type::Number, &index)) return ParseError::Capacity;
    if (peek() == '-') ++pos_;
    if (at_end()) return ParseError::Syntax;
    if (peek() == '0') {
      ++pos_;
    } else if (!digits()) {
      return ParseError::Syntax;
    }
    if (!at_end() && peek() == '.') {
      ++pos_;
      if (!digits()) return ParseError::Syntax;
    }
    if (!at_end() && (peek() == 'e' || peek() == 'E')) {
      ++pos_;
      if (!at_end() && (peek() == '+' || peek() == '-')) ++pos_;
      if (!digits()) return ParseError::Syntax;
    }
    pool_[index].end = static_cast<uint32_t>(pos_);
    return ParseError::None;
  }

  Token* pool_;
  uint32_t capacity_;
  const char* text_ = nullptr;
  size_t len_ = 0;
  size_t pos_ = 0;
  uint32_t used_ = 0;
};

inline const Token* Value::token() const { return ok() ? &doc_->token(index_) : nullptr; }

inline const char* Value::raw() const { return doc_->text() + token()->start; }

inline Type Value::type() const { return ok() ? token()->type : Type::Null; }

inline uint32_t Value::size() const {
  if (!is_array() && !is_object()) return 0;
  return token()->count;
}

inline Value Value::at(uint32_t i) const {
  if (i >= size()) return Value();
  uint32_t child = index_ + 1;
  const bool object = is_object();
  for (uint32_t k = 0; k < i; ++k) {
    if (object) child = doc_->token(child).next;  // past the key
    child = doc_->token(child).next;
  }
  if (object) child = doc_->token(child).next;
  return Value(doc_, child);
}

inline Value Value::key_at(uint32_t i) const {
  if (!is_object() || i >= size()) return Value();
  uint32_t child = index_ + 1;
  for (uint32_t k = 0; k < i; ++k) {
    child = doc_->token(child).next;
    child = doc_->token(child).next;
  }
  return Value(doc_, child);
}

inline Value Value::get(const char* key) const {
  if (!is_object()) return Value();
  uint32_t child = index_ + 1;
  for (uint32_t k = 0; k < token()->count; ++k) {
    const Value name(doc_, child);
    const uint32_t value = doc_->token(child).next;
    if (name.equals(key)) return Value(doc_, value);
    child = doc_->token(value).next;
  }
  return Value();
}

inline bool Value::get_bool(bool* out) const {
  if (!is_bool()) return false;
  *out = type() == Type::True;
  return true;
}

inline bool Value::get_int(int64_t* out) const {
  if (!is_number()) return false;
  const char* p = raw();
  const char* end = doc_->text() + token()->end;
  const bool negative = *p == '-';
  if (negative) ++p;
  // The bound is the magnitude of INT64_MIN, so the most negative value parses too.
  const uint64_t limit = negative ? (static_cast<uint64_t>(INT64_MAX) + 1u)
                                  : static_cast<uint64_t>(INT64_MAX);
  uint64_t value = 0;
  for (; p < end; ++p) {
    if (*p < '0' || *p > '9') return false;
    const uint64_t digit = static_cast<uint64_t>(*p - '0');
    if (value > (limit - digit) / 10u) return false;
    value = value * 10u + digit;
  }
  if (negative) {
    *out = value == static_cast<uint64_t>(INT64_MAX) + 1u ? INT64_MIN
                                                          : -static_cast<int64_t>(value);
  } else {
    *out = static_cast<int64_t>(value);
  }
  return true;
}

inline bool Value::get_double(double* out) const {
  if (!is_number()) return false;
  const size_t len = token()->end - token()->start;
  char buf[64];
  if (len >= sizeof(buf)) return false;
  std::memcpy(buf, raw(), len);
  buf[len] = '\0';
  char* end = nullptr;
  const double value = std::strtod(buf, &end);
  if (end != buf + len || !std::isfinite(value)) return false;
  *out = value;
  return true;
}

inline bool Value::get_string(char* out, size_t cap, size_t* len) const {
  if (!is_string() || cap == 0) return false;
  detail::StringDecoder decoder(raw(), token()->end - token()->start);
  size_t n = 0;
  uint8_t byte = 0;
  while (decoder.next(&byte)) {
    if (n + 1 >= cap) return false;
    out[n++] = static_cast<char>(byte);
  }
  out[n] = '\0';
  if (len != nullptr) *len = n;
  return true;
}

inline bool Value::equals(const char* text) const {
  if (!is_string()) return false;
  detail::StringDecoder decoder(raw(), token()->end - token()->start);
  const unsigned char* p = reinterpret_cast<const unsigned char*>(text);
  uint8_t byte = 0;
  while (decoder.next(&byte)) {
    if (*p == '\0' || *p != byte) return false;
    ++p;
  }
  return *p == '\0';
}

// Writes JSON into a caller-given buffer, always NUL-terminated. A value that does not fit, or a
// number JSON cannot carry, marks the writer failed; the caller checks `ok()` once at the end.
// Commas are placed by the writer, so a builder only states keys and values in order.
class Writer {
 public:
  Writer(char* buf, size_t cap) : buf_(buf), cap_(cap) {
    if (cap_ == 0) {
      ok_ = false;
    } else {
      buf_[0] = '\0';
    }
  }

  bool ok() const { return ok_; }
  size_t size() const { return len_; }
  const char* data() const { return buf_; }
  void fail() { ok_ = false; }

  Writer& begin_object() {
    separate();
    put('{');
    comma_ = false;
    return *this;
  }
  Writer& end_object() {
    put('}');
    comma_ = true;
    return *this;
  }
  Writer& begin_array() {
    separate();
    put('[');
    comma_ = false;
    return *this;
  }
  Writer& end_array() {
    put(']');
    comma_ = true;
    return *this;
  }
  Writer& key(const char* name) {
    separate();
    quoted(name, std::strlen(name));
    put(':');
    comma_ = false;
    return *this;
  }
  Writer& string(const char* text) { return string(text, std::strlen(text)); }
  Writer& string(const char* text, size_t len) {
    separate();
    quoted(text, len);
    comma_ = true;
    return *this;
  }
  Writer& integer(int64_t value) {
    char tmp[24];
    std::snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(value));
    return raw_value(tmp);
  }
  // The shortest text that reads back as `value`, always with a fraction or an exponent so a
  // strict reader sees a float.
  Writer& number(double value) {
    if (!std::isfinite(value)) {
      ok_ = false;
      return *this;
    }
    char tmp[40];
    for (int precision = 1; precision <= 17; ++precision) {
      std::snprintf(tmp, sizeof(tmp), "%.*g", precision, value);
      if (std::strtod(tmp, nullptr) == value) break;
    }
    return raw_value(as_float_text(tmp, sizeof(tmp)));
  }
  // As `number`, shortest for a single-precision value, so 0.87f writes as 0.87.
  Writer& number_float(float value) {
    if (!std::isfinite(value)) {
      ok_ = false;
      return *this;
    }
    char tmp[40];
    for (int precision = 1; precision <= 9; ++precision) {
      std::snprintf(tmp, sizeof(tmp), "%.*g", precision, static_cast<double>(value));
      if (std::strtof(tmp, nullptr) == value) break;
    }
    return raw_value(as_float_text(tmp, sizeof(tmp)));
  }
  Writer& boolean(bool value) { return raw_value(value ? "true" : "false"); }
  Writer& null() { return raw_value("null"); }

 private:
  static const char* as_float_text(char* tmp, size_t cap) {
    if (std::strpbrk(tmp, ".eEn") == nullptr) {
      const size_t len = std::strlen(tmp);
      if (len + 3 <= cap) std::memcpy(tmp + len, ".0", 3);
    }
    return tmp;
  }

  Writer& raw_value(const char* text) {
    separate();
    puts(text);
    comma_ = true;
    return *this;
  }

  void separate() {
    if (comma_) put(',');
  }

  void put(char c) {
    if (!ok_ || len_ + 1 >= cap_) {
      ok_ = false;
      return;
    }
    buf_[len_++] = c;
    buf_[len_] = '\0';
  }

  void puts(const char* text) {
    for (; *text != '\0'; ++text) put(*text);
  }

  void quoted(const char* text, size_t len) {
    static const char kHex[] = "0123456789abcdef";
    put('"');
    for (size_t i = 0; i < len; ++i) {
      const unsigned char c = static_cast<unsigned char>(text[i]);
      switch (c) {
        case '"': puts("\\\""); break;
        case '\\': puts("\\\\"); break;
        case '\n': puts("\\n"); break;
        case '\r': puts("\\r"); break;
        case '\t': puts("\\t"); break;
        case '\b': puts("\\b"); break;
        case '\f': puts("\\f"); break;
        default:
          if (c < 0x20) {
            puts("\\u00");
            put(kHex[c >> 4]);
            put(kHex[c & 0x0F]);
          } else {
            put(static_cast<char>(c));
          }
      }
    }
    put('"');
  }

  char* buf_;
  size_t cap_;
  size_t len_ = 0;
  bool ok_ = true;
  bool comma_ = false;
};

}  // namespace json
}  // namespace mist_core
