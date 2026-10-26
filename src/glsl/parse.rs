//! Read a Hydra chain written as text.
//!
//! The score realm hands over a *recording* - structured nodes - because it
//! watches a score being evaluated. A snippet from the shelf is not evaluated
//! by anything: it is the text a reader is about to copy, and it has to become
//! the same nodes.
//!
//! Hydra's chain grammar is small enough to read directly, which is what
//! makes this a parser rather than an interpreter:
//!
//! ```text
//! chain  := call ('.' call)*
//! call   := name '(' args? ')'
//! args   := arg (',' arg)*
//! arg    := number | chain | name | callback
//! ```
//!
//! A callback is an arrow function or a parenthesized expression. It is kept
//! as source text, and the composer compiles it or refuses it. A string or an
//! array is refused by name rather than mis-parsed, because a snippet that
//! silently renders the wrong thing is worse than one that says it cannot.

use crate::program::{
    HydraCall, HydraNode, MAX_HYDRA_ARGS, MAX_HYDRA_CALLS, MAX_HYDRA_DEPTH, MAX_HYDRA_NAME_BYTES,
    MAX_HYDRA_NODES, MAX_HYDRA_SOURCE_BYTES,
};

#[derive(Debug, PartialEq)]
pub enum ParseError {
    /// Something the grammar has no room for, with what was found.
    Unexpected { at: usize, found: String },
    /// The text ran out mid-chain.
    Ended,
    /// Valid Hydra, but beyond what this reader does: a string, an array, a
    /// number literal a shader float cannot hold.
    Unsupported(String),
    /// Input that is valid in shape but exceeds a synchronous parser bound.
    Limit { what: &'static str, limit: usize },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unexpected { at, found } => {
                write!(f, "unexpected `{found}` at byte {at}")
            }
            Self::Ended => write!(f, "the chain ends before it is finished"),
            Self::Unsupported(what) => write!(f, "{what}"),
            Self::Limit { what, limit } => {
                write!(f, "the Hydra snippet exceeds the {limit} {what} limit")
            }
        }
    }
}

/// Read one chain. Trailing whitespace, comments and a trailing `;` are fine.
pub fn parse_chain(text: &str) -> Result<HydraNode, ParseError> {
    if text.len() > MAX_HYDRA_SOURCE_BYTES {
        return Err(ParseError::Limit {
            what: "byte",
            limit: MAX_HYDRA_SOURCE_BYTES,
        });
    }
    let mut reader = Reader {
        bytes: text.as_bytes(),
        at: 0,
        nodes: 0,
    };
    let node = reader.chain(0)?;
    reader.space();
    if reader.peek() == Some(b';') {
        reader.at += 1;
        reader.space();
    }
    if reader.at < reader.bytes.len() {
        return Err(ParseError::Unexpected {
            at: reader.at,
            found: reader.rest_summary(),
        });
    }
    Ok(node)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    nodes: usize,
}

impl Reader<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn rest_summary(&self) -> String {
        String::from_utf8_lossy(&self.bytes[self.at..self.bytes.len().min(self.at + 24)]).into()
    }

    /// Whitespace, and `//` comments, which snippets carry.
    fn space(&mut self) {
        loop {
            while self.peek().is_some_and(|b| b.is_ascii_whitespace()) {
                self.at += 1;
            }
            if self.bytes[self.at..].starts_with(b"//") {
                while self.peek().is_some_and(|b| b != b'\n') {
                    self.at += 1;
                }
                continue;
            }
            return;
        }
    }

    fn name(&mut self) -> Result<String, ParseError> {
        self.space();
        let start = self.at;
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'$')
        {
            self.at += 1;
        }
        if start == self.at {
            return Err(if self.at >= self.bytes.len() {
                ParseError::Ended
            } else {
                ParseError::Unexpected {
                    at: self.at,
                    found: self.rest_summary(),
                }
            });
        }
        if self.at - start > MAX_HYDRA_NAME_BYTES {
            return Err(ParseError::Limit {
                what: "name-byte",
                limit: MAX_HYDRA_NAME_BYTES,
            });
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned())
    }

    fn chain(&mut self, depth: usize) -> Result<HydraNode, ParseError> {
        self.depth(depth)?;
        let head = self.name()?;
        let args = self.args(depth + 1)?;
        let mut calls = Vec::new();
        loop {
            self.space();
            if self.peek() != Some(b'.') {
                break;
            }
            if calls.len() >= MAX_HYDRA_CALLS {
                return Err(ParseError::Limit {
                    what: "chained-call",
                    limit: MAX_HYDRA_CALLS,
                });
            }
            // A decimal point belongs to a number, not to a method.
            self.at += 1;
            let method = self.name()?;
            let args = self.args(depth + 1)?;
            calls.push(HydraCall { method, args });
        }
        self.node(HydraNode::Chain { head, args, calls })
    }

    /// `(` args `)`, which every call has even when it is empty.
    fn args(&mut self, depth: usize) -> Result<Vec<HydraNode>, ParseError> {
        self.space();
        if self.peek() != Some(b'(') {
            return Err(ParseError::Unexpected {
                at: self.at,
                found: self.rest_summary(),
            });
        }
        self.at += 1;
        let mut args = Vec::new();
        self.space();
        if self.peek() == Some(b')') {
            self.at += 1;
            return Ok(args);
        }
        loop {
            if args.len() >= MAX_HYDRA_ARGS {
                return Err(ParseError::Limit {
                    what: "argument",
                    limit: MAX_HYDRA_ARGS,
                });
            }
            args.push(self.arg(depth)?);
            self.space();
            match self.peek() {
                Some(b')') => {
                    self.at += 1;
                    return Ok(args);
                }
                Some(b',') => {
                    self.at += 1;
                    self.space();
                    // Modern JavaScript permits one trailing comma in call
                    // arguments, and pasted Hydra sketches use that grammar.
                    if self.peek() == Some(b')') {
                        self.at += 1;
                        return Ok(args);
                    }
                }
                None => return Err(ParseError::Ended),
                _ => {
                    return Err(ParseError::Unexpected {
                        at: self.at,
                        found: self.rest_summary(),
                    });
                }
            }
        }
    }

    fn arg(&mut self, depth: usize) -> Result<HydraNode, ParseError> {
        self.depth(depth)?;
        self.space();
        match self.peek() {
            Some(b'"' | b'\'' | b'`') => Err(ParseError::Unsupported(
                "a snippet with a string in it is not something this reads".into(),
            )),
            Some(b'[') => Err(ParseError::Unsupported(
                "a snippet with an array in it is not something this reads".into(),
            )),
            // `() => …` - kept as written; the composer decides whether it
            // is an expression it can turn into GLSL.
            Some(b'(') => self.callback(),
            Some(byte) if byte.is_ascii_digit() || byte == b'-' || byte == b'.' => self.number(),
            Some(_) => {
                // A name: either a nested chain, or a bare global like `o0`.
                let start = self.at;
                let name = self.name()?;
                self.space();
                if self.peek() == Some(b'(') {
                    self.at = start;
                    return self.chain(depth);
                }
                if self.peek() == Some(b'=') {
                    self.at = start;
                    return self.callback();
                }
                self.node(HydraNode::Global { name })
            }
            None => Err(ParseError::Ended),
        }
    }

    /// A `() => …` argument, taken as far as its argument ends.
    ///
    /// Balanced brackets rather than a grammar, because what is inside is the
    /// composer's business: this only has to know where it stops.
    fn callback(&mut self) -> Result<HydraNode, ParseError> {
        let start = self.at;
        let mut depth = 0usize;
        while let Some(byte) = self.peek() {
            match byte {
                b'(' | b'[' | b'{' => {
                    depth = depth.checked_add(1).ok_or(ParseError::Limit {
                        what: "callback-depth",
                        limit: MAX_HYDRA_DEPTH,
                    })?;
                    if depth > MAX_HYDRA_DEPTH {
                        return Err(ParseError::Limit {
                            what: "callback-depth",
                            limit: MAX_HYDRA_DEPTH,
                        });
                    }
                }
                b')' | b']' | b'}' if depth == 0 => break,
                b')' | b']' | b'}' => depth -= 1,
                b',' if depth == 0 => break,
                _ => {}
            }
            self.at += 1;
        }
        let src = String::from_utf8_lossy(&self.bytes[start..self.at])
            .trim()
            .to_owned();
        if src.is_empty() {
            return Err(ParseError::Ended);
        }
        self.node(HydraNode::Source { src })
    }

    fn number(&mut self) -> Result<HydraNode, ParseError> {
        let start = self.at;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.at += 1;
        }
        let is_hex =
            self.bytes[self.at..].starts_with(b"0x") || self.bytes[self.at..].starts_with(b"0X");
        let value = if is_hex {
            self.at += 2;
            let digits_start = self.at;
            while self.peek().is_some_and(|b| b.is_ascii_hexdigit()) {
                self.at += 1;
            }
            if self.at == digits_start {
                return Err(self.malformed_number(start));
            }
            let digits = &self.bytes[digits_start..self.at];
            let significant = &digits[digits
                .iter()
                .position(|&b| b != b'0')
                .unwrap_or(digits.len())..];
            // More than 128 significant bits is beyond every finite shader
            // float. Leading zeroes do not change the value or this bound.
            if significant.len() > 32 {
                f64::INFINITY
            } else if significant.is_empty() {
                0.0
            } else {
                let hex =
                    std::str::from_utf8(significant).map_err(|_| self.malformed_number(start))?;
                u128::from_str_radix(hex, 16).map_err(|_| self.malformed_number(start))? as f64
            }
        } else {
            let mut digits = 0;
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.at += 1;
                digits += 1;
            }
            if self.peek() == Some(b'.') {
                self.at += 1;
                while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.at += 1;
                    digits += 1;
                }
            }
            if digits == 0 {
                return Err(self.malformed_number(start));
            }
            if self.peek().is_some_and(|b| b == b'e' || b == b'E') {
                self.at += 1;
                if self.peek().is_some_and(|b| b == b'+' || b == b'-') {
                    self.at += 1;
                }
                let exponent_start = self.at;
                while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.at += 1;
                }
                if self.at == exponent_start {
                    return Err(self.malformed_number(start));
                }
            }
            let decimal = std::str::from_utf8(&self.bytes[start..self.at])
                .map_err(|_| self.malformed_number(start))?;
            decimal
                .parse::<f64>()
                .map_err(|_| self.malformed_number(start))?
        };
        let text = String::from_utf8_lossy(&self.bytes[start..self.at]);
        let value = if is_hex && negative { -value } else { value };
        // A GLSL float is an f32. A literal past its range lexes as infinity
        // in the composed shader and fails every pipeline build: per frame,
        // uncached, and with no name in the error. Examples are `1e39`, a
        // forty-digit integer, and `1e999`, which Rust's f64 parse already
        // overflows to infinity. The composer refuses such a number too, but
        // only this reader still has the literal as written, so it refuses
        // the number here and names it.
        if !crate::glsl::compose::shader_float(value) {
            return Err(ParseError::Unsupported(crate::glsl::compose::out_of_range(
                &text,
            )));
        }
        self.node(HydraNode::Number { v: value })
    }

    fn malformed_number(&self, start: usize) -> ParseError {
        ParseError::Unexpected {
            at: start,
            found: String::from_utf8_lossy(&self.bytes[start..self.bytes.len().min(start + 24)])
                .into_owned(),
        }
    }

    fn depth(&self, depth: usize) -> Result<(), ParseError> {
        if depth > MAX_HYDRA_DEPTH {
            return Err(ParseError::Limit {
                what: "nesting-depth",
                limit: MAX_HYDRA_DEPTH,
            });
        }
        Ok(())
    }

    fn node(&mut self, node: HydraNode) -> Result<HydraNode, ParseError> {
        self.nodes = self.nodes.checked_add(1).ok_or(ParseError::Limit {
            what: "node",
            limit: MAX_HYDRA_NODES,
        })?;
        if self.nodes > MAX_HYDRA_NODES {
            return Err(ParseError::Limit {
                what: "node",
                limit: MAX_HYDRA_NODES,
            });
        }
        Ok(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod numeric_literals {
        use super::*;

        #[test]
        fn snippet_numbers_accept_signed_exponents_and_hexadecimal_literals() {
            for (source, value) in [
                ("osc(1e-5)", 1e-5),
                ("osc(1E+3)", 1000.0),
                ("osc(-2.5e-2)", -0.025),
                ("osc(.5e+1)", 5.0),
                ("osc(1.e-1)", 0.1),
                ("osc(0xff)", 255.0),
                ("osc(0X1e)", 30.0),
                ("osc(-0x10)", -16.0),
                ("osc(0x000000ff)", 255.0),
                ("osc(0x0000000000000000000000000000000000000000ff)", 255.0),
                ("osc(0x000000000000000000000000000000000000000000)", 0.0),
                (
                    "osc(0xffffff00000000000000000000000000)",
                    f64::from(f32::MAX),
                ),
            ] {
                let Ok(HydraNode::Chain { args, .. }) = parse_chain(source) else {
                    panic!("valid snippet number should parse: `{source}`");
                };
                assert_eq!(args, vec![HydraNode::Number { v: value }], "{source}");
            }
        }

        #[test]
        fn malformed_numeric_literals_remain_syntax_errors() {
            for source in [
                "osc(1e+)",
                "osc(1e-)",
                "osc(1e--5)",
                "osc(1e+2.5)",
                "osc(0x)",
                "osc(0xGG)",
                "osc(0x1G)",
                "osc(0x1.2)",
                "osc(0x1p2)",
            ] {
                assert!(
                    matches!(parse_chain(source), Err(ParseError::Unexpected { .. })),
                    "expected a syntax error for `{source}`"
                );
            }
        }

        #[test]
        fn hexadecimal_literals_keep_the_shader_float_range_limit() {
            for literal in [
                "0xffffffffffffffffffffffffffffffff",
                "0x100000000000000000000000000000000",
                "-0x100000000000000000000000000000000",
            ] {
                let error = parse_chain(&format!("osc({literal})"))
                    .expect_err("the shader cannot represent this finite integer");
                assert!(matches!(error, ParseError::Unsupported(_)));
                let message = error.to_string();
                assert!(message.contains(literal), "{message}");
                assert!(
                    message.contains("is out of range for a shader float"),
                    "{message}"
                );
            }
        }

        #[test]
        fn signed_exponents_and_hex_compose_in_nested_and_chained_arguments() {
            let from_literals =
                parse_chain("osc(1e-5).modulate(noise(0x10)).rotate(-2.5e-2).out()")
                    .expect("numeric snippets parse");
            let decimal = parse_chain("osc(0.00001).modulate(noise(16)).rotate(-0.025).out()")
                .expect("decimal snippet parses");
            assert_eq!(from_literals, decimal);
            crate::glsl::compose(&from_literals, "highp").expect("numeric snippets compose");
        }
    }

    #[test]
    fn source_bytes_arguments_calls_and_names_are_bounded_before_growth() {
        assert!(matches!(
            parse_chain(&"x".repeat(MAX_HYDRA_SOURCE_BYTES + 1)),
            Err(ParseError::Limit { what: "byte", .. })
        ));

        let arguments = std::iter::repeat_n("0", MAX_HYDRA_ARGS + 1)
            .collect::<Vec<_>>()
            .join(",");
        assert!(matches!(
            parse_chain(&format!("solid({arguments})")),
            Err(ParseError::Limit {
                what: "argument",
                ..
            })
        ));

        let calls = format!("osc(){}", ".rotate()".repeat(MAX_HYDRA_CALLS + 1));
        assert!(matches!(
            parse_chain(&calls),
            Err(ParseError::Limit {
                what: "chained-call",
                ..
            })
        ));

        let long_name = "x".repeat(MAX_HYDRA_NAME_BYTES + 1);
        assert!(matches!(
            parse_chain(&format!("{long_name}()")),
            Err(ParseError::Limit {
                what: "name-byte",
                ..
            })
        ));
    }

    #[test]
    fn arguments_require_real_separators_and_allow_one_trailing_comma() {
        for malformed in [
            "osc(1 2)",
            "osc(,1)",
            "osc(1,,2)",
            "osc(1,,)",
            "osc(,)",
            "osc().rotate(1 2)",
        ] {
            assert!(parse_chain(malformed).is_err(), "accepted `{malformed}`");
        }

        parse_chain("osc(1, 2,)").expect("JavaScript permits a trailing argument comma");
        parse_chain("osc().rotate(1,)").expect("method calls permit it too");
    }

    #[test]
    fn nested_chains_stop_at_the_protocol_depth_without_recursing_past_it() {
        let mut boundary = "osc()".to_owned();
        for _ in 0..MAX_HYDRA_DEPTH {
            boundary = format!("src({boundary})");
        }
        parse_chain(&boundary).expect("the protocol depth boundary is accepted");

        let too_deep = format!("src({boundary})");
        assert!(matches!(
            parse_chain(&too_deep),
            Err(ParseError::Limit {
                what: "nesting-depth",
                ..
            })
        ));
    }

    #[test]
    fn callback_brackets_and_total_nodes_have_explicit_ceilings() {
        let callback = format!(
            "osc(() => {}0{})",
            "(".repeat(MAX_HYDRA_DEPTH + 1),
            ")".repeat(MAX_HYDRA_DEPTH + 1)
        );
        assert!(matches!(
            parse_chain(&callback),
            Err(ParseError::Limit {
                what: "callback-depth",
                ..
            })
        ));

        let mut reader = Reader {
            bytes: b"",
            at: 0,
            nodes: MAX_HYDRA_NODES,
        };
        assert!(matches!(
            reader.node(HydraNode::Null),
            Err(ParseError::Limit { what: "node", .. })
        ));
    }

    #[test]
    fn a_literal_no_shader_float_can_hold_is_refused_by_name() {
        // Overflow of every kind: an exponent past f64's range, digits past
        // it with no exponent at all, and - the common case - values finite
        // as f64 that an f32, which is what a GLSL float is, cannot hold.
        let past_f64 = format!("osc(1{})", "0".repeat(309));
        let forty_digits = format!("osc(1{})", "0".repeat(39));
        for overflowing in [
            "osc(1e999)",
            "osc(-1e999)",
            "osc(1E999)",
            "osc(1e309)",
            past_f64.as_str(),
            "osc(1e300)",
            "osc(-1e300)",
            "osc(1e39)",
            forty_digits.as_str(),
            // Just past the point where an f32 rounds to infinity.
            "osc(3.4028236e38)",
            "osc().rotate(1e39)",
        ] {
            let error = parse_chain(overflowing)
                .expect_err("GLSL has no literal for an infinite float")
                .to_string();
            assert!(
                error.contains("is out of range for a shader float"),
                "refused by name: `{overflowing}` said {error}"
            );
        }

        for (source, literal) in [
            ("osc(1e999)", "`1e999`"),
            ("osc(-1e999)", "`-1e999`"),
            ("osc(1e39)", "`1e39`"),
            ("osc(-1e300)", "`-1e300`"),
        ] {
            let error = parse_chain(source).expect_err("refused").to_string();
            assert!(
                error.contains(literal),
                "names {literal} as written: {error}"
            );
        }
    }

    #[test]
    fn literals_a_shader_float_can_hold_stay_legal_numbers() {
        for (source, value) in [
            ("osc(3e38)", 3e38),
            ("osc(-3e38)", -3e38),
            // `f32::MAX`'s own shortest spelling. As an f64 it is a little
            // above `f32::MAX`, but it rounds to it, as naga's lexer does.
            ("osc(3.4028235e38)", 3.4028235e38),
            ("osc(0.5)", 0.5),
            ("osc(-2.5e3)", -2500.0),
            ("osc(12)", 12.0),
        ] {
            let Ok(HydraNode::Chain { args, .. }) = parse_chain(source) else {
                panic!("`{source}` is a literal a shader float can hold");
            };
            assert_eq!(args, vec![HydraNode::Number { v: value }], "{source}");
        }
    }
}
