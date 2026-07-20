#!/usr/bin/env python3
"""Validate the deliberately small Lean grammar used by formal snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys


class SnapshotSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    column: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    line = 1
    column = 1

    def advance(text: str) -> None:
        nonlocal line, column
        lines = text.split("\n")
        if len(lines) == 1:
            column += len(text)
        else:
            line += len(lines) - 1
            column = len(lines[-1]) + 1

    while index < len(source):
        if source[index].isspace():
            advance(source[index])
            index += 1
            continue
        if source.startswith("--", index):
            end = source.find("\n", index)
            if end == -1:
                advance(source[index:])
                break
            advance(source[index:end])
            index = end
            continue
        if source.startswith("/-", index):
            start_line, start_column = line, column
            depth = 1
            end = index + 2
            while end < len(source) and depth != 0:
                if source.startswith("/-", end):
                    depth += 1
                    end += 2
                elif source.startswith("-/", end):
                    depth -= 1
                    end += 2
                else:
                    end += 1
            if depth != 0:
                raise SnapshotSyntaxError(
                    f"{start_line}:{start_column}: unterminated block comment"
                )
            advance(source[index:end])
            index = end
            continue

        token_line, token_column = line, column
        if source[index] == '"':
            end = index + 1
            while end < len(source):
                if source[end] == "\\":
                    end += 2
                    continue
                if source[end] == '"':
                    end += 1
                    break
                if source[end] == "\n":
                    raise SnapshotSyntaxError(
                        f"{token_line}:{token_column}: newline in string literal"
                    )
                end += 1
            else:
                raise SnapshotSyntaxError(
                    f"{token_line}:{token_column}: unterminated string literal"
                )
            text = source[index:end]
            tokens.append(Token("string", text, token_line, token_column))
            advance(text)
            index = end
            continue

        if source.startswith(":=", index):
            tokens.append(Token("symbol", ":=", token_line, token_column))
            advance(":=")
            index += 2
            continue

        byte = source[index]
        if byte.isalpha() or byte == "_":
            end = index + 1
            while end < len(source) and (
                source[end].isalnum() or source[end] in "_'?!"
            ):
                end += 1
            text = source[index:end]
            tokens.append(Token("identifier", text, token_line, token_column))
            advance(text)
            index = end
            continue
        if byte.isdigit():
            end = index + 1
            while end < len(source) and source[end].isdigit():
                end += 1
            text = source[index:end]
            tokens.append(Token("number", text, token_line, token_column))
            advance(text)
            index = end
            continue
        if byte in "()[],:×.":
            tokens.append(Token("symbol", byte, token_line, token_column))
            advance(byte)
            index += 1
            continue
        raise SnapshotSyntaxError(
            f"{token_line}:{token_column}: token {byte!r} is outside the snapshot grammar"
        )

    tokens.append(Token("eof", "", line, column))
    return tokens


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0
        self.definition_count = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def fail(self, message: str) -> SnapshotSyntaxError:
        token = self.current
        return SnapshotSyntaxError(f"{token.line}:{token.column}: {message}")

    def accept(self, text: str) -> bool:
        if self.current.text != text:
            return False
        self.index += 1
        return True

    def expect(self, text: str) -> None:
        if not self.accept(text):
            raise self.fail(f"expected {text!r}, found {self.current.text!r}")

    def expect_identifier(self) -> str:
        if self.current.kind != "identifier":
            raise self.fail(f"expected identifier, found {self.current.text!r}")
        text = self.current.text
        self.index += 1
        return text

    def parse_module(self) -> None:
        self.expect("namespace")
        self.expect("Ora")
        self.expect(".")
        self.expect("Generated")
        while self.current.text == "def":
            self.parse_definition()
        if self.definition_count == 0:
            raise self.fail("snapshot must define at least one data value")
        self.expect("end")
        self.expect("Ora")
        self.expect(".")
        self.expect("Generated")
        if self.current.kind != "eof":
            raise self.fail(f"unexpected trailing command {self.current.text!r}")

    def parse_definition(self) -> None:
        self.expect("def")
        self.expect_identifier()
        self.expect(":")
        self.parse_type()
        self.expect(":=")
        self.parse_literal()
        self.definition_count += 1

    def parse_type(self) -> None:
        if self.accept("List") or self.accept("Option"):
            self.parse_type()
            return
        if self.current.text in {"String", "Nat", "Bool"}:
            self.index += 1
            return
        if self.accept("("):
            self.parse_type()
            if not self.accept("×"):
                raise self.fail("parenthesized snapshot types must be products")
            self.parse_type()
            while self.accept("×"):
                self.parse_type()
            self.expect(")")
            return
        raise self.fail(f"type token {self.current.text!r} is outside the snapshot grammar")

    def parse_literal(self) -> None:
        if self.current.kind in {"number", "string"}:
            self.index += 1
            return
        if self.current.text in {"true", "false", "none"}:
            self.index += 1
            return
        if self.accept("some"):
            self.parse_literal()
            return
        if self.accept("["):
            if self.accept("]"):
                return
            self.parse_literal()
            while self.accept(","):
                self.parse_literal()
            self.expect("]")
            return
        if self.accept("("):
            self.parse_literal()
            if not self.accept(","):
                raise self.fail("parenthesized snapshot values must be tuples")
            self.parse_literal()
            while self.accept(","):
                self.parse_literal()
            self.expect(")")
            return
        raise self.fail(
            f"value token {self.current.text!r} is not a data literal"
        )


def validate_snapshot(source: str) -> None:
    Parser(tokenize(source)).parse_module()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <snapshot.lean>", file=sys.stderr)
        return 2
    path = pathlib.Path(argv[1])
    try:
        validate_snapshot(path.read_text(encoding="utf-8"))
    except (OSError, SnapshotSyntaxError) as error:
        print(f"{path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
