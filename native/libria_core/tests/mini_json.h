#pragma once
// Parser JSON mínimo, só para ler as fixtures nos testes (sem dependências).

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_json {

class Value {
public:
    enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

    Type type = Type::kNull;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    bool is_null() const { return type == Type::kNull; }
    bool has(const std::string& key) const { return object.count(key) != 0; }

    const Value& operator[](const std::string& key) const {
        require(Type::kObject, "objeto");
        const auto found = object.find(key);
        if (found == object.end()) throw std::out_of_range("chave ausente: " + key);
        return found->second;
    }
    const Value& operator[](std::size_t index) const {
        require(Type::kArray, "array");
        return array.at(index);
    }
    std::size_t size() const { return type == Type::kArray ? array.size() : object.size(); }
    double num() const { require(Type::kNumber, "número"); return number; }
    bool flag() const { require(Type::kBool, "booleano"); return boolean; }
    const std::string& str() const { require(Type::kString, "string"); return string; }

private:
    void require(Type expected, const char* name) const {
        if (type != expected) throw std::runtime_error(std::string("valor JSON não é ") + name);
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : p_(text.data()), end_(text.data() + text.size()) {}

    Value parse() {
        Value value = parse_value();
        skip_space();
        if (p_ != end_) fail("conteúdo extra após o JSON");
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) { throw std::runtime_error("JSON: " + message); }

    void skip_space() {
        while (p_ != end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
    }

    bool consume(const char* literal) {
        const char* q = p_;
        for (const char* l = literal; *l; ++l, ++q) {
            if (q == end_ || *q != *l) return false;
        }
        p_ = q;
        return true;
    }

    Value parse_value() {
        skip_space();
        if (p_ == end_) fail("fim inesperado");
        Value value;
        switch (*p_) {
            case 'n': if (!consume("null")) fail("literal inválido"); return value;
            case 't': if (!consume("true")) fail("literal inválido"); value.type = Value::Type::kBool; value.boolean = true; return value;
            case 'f': if (!consume("false")) fail("literal inválido"); value.type = Value::Type::kBool; return value;
            case '"': value.type = Value::Type::kString; value.string = parse_string(); return value;
            case '[': return parse_array();
            case '{': return parse_object();
            default: return parse_number();
        }
    }

    Value parse_number() {
        const char* start = p_;
        while (p_ != end_ && (std::isdigit(static_cast<unsigned char>(*p_)) || *p_ == '-' || *p_ == '+' ||
                              *p_ == '.' || *p_ == 'e' || *p_ == 'E')) {
            ++p_;
        }
        if (start == p_) fail("valor inválido");
        const std::string token(start, p_);
        char* parsed_end = nullptr;
        Value value;
        value.type = Value::Type::kNumber;
        value.number = std::strtod(token.c_str(), &parsed_end);  // locale "C" no teste
        if (parsed_end != token.c_str() + token.size()) fail("número inválido: " + token);
        return value;
    }

    static void append_utf8(std::string& out, unsigned code) {
        if (code < 0x80) { out += static_cast<char>(code); }
        else if (code < 0x800) { out += static_cast<char>(0xC0 | (code >> 6)); out += static_cast<char>(0x80 | (code & 0x3F)); }
        else { out += static_cast<char>(0xE0 | (code >> 12)); out += static_cast<char>(0x80 | ((code >> 6) & 0x3F)); out += static_cast<char>(0x80 | (code & 0x3F)); }
    }

    std::string parse_string() {
        ++p_;  // aspas de abertura
        std::string out;
        while (true) {
            if (p_ == end_) fail("string sem fim");
            const char c = *p_++;
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (p_ == end_) fail("escape sem fim");
            const char e = *p_++;
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (end_ - p_ < 4) fail("\\u incompleto");
                    append_utf8(out, static_cast<unsigned>(std::stoul(std::string(p_, p_ + 4), nullptr, 16)));
                    p_ += 4;
                    break;
                }
                default: fail("escape inválido");
            }
        }
    }

    Value parse_array() {
        ++p_;
        Value value;
        value.type = Value::Type::kArray;
        skip_space();
        if (p_ != end_ && *p_ == ']') { ++p_; return value; }
        while (true) {
            value.array.push_back(parse_value());
            skip_space();
            if (p_ == end_) fail("array sem fim");
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == ']') { ++p_; return value; }
            fail("esperado , ou ]");
        }
    }

    Value parse_object() {
        ++p_;
        Value value;
        value.type = Value::Type::kObject;
        skip_space();
        if (p_ != end_ && *p_ == '}') { ++p_; return value; }
        while (true) {
            skip_space();
            if (p_ == end_ || *p_ != '"') fail("esperada chave");
            std::string key = parse_string();
            skip_space();
            if (p_ == end_ || *p_ != ':') fail("esperado :");
            ++p_;
            value.object[key] = parse_value();
            skip_space();
            if (p_ == end_) fail("objeto sem fim");
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == '}') { ++p_; return value; }
            fail("esperado , ou }");
        }
    }

    const char* p_;
    const char* end_;
};

inline Value parse_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("não foi possível abrir " + path);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return Parser(buffer.str()).parse();
}

}  // namespace mini_json
