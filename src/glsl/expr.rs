//! Compile the small expressions a Hydra sketch writes as callbacks.
//!
//! Callback expressions such as `osc(() => time * 0.05)` are compiled into
//! GLSL. The shader reads `time` and audio bands from uniforms; the renderer
//! does not execute JavaScript callbacks.
//!
//! Supported expressions are limited to arithmetic, parentheses, a handful of
//! `Math` functions, `time`, and `a.fft[n]`. Unsupported expressions are
//! rejected rather than silently changing the sketch's behavior.

/// The first `vec4` in the fixed-capacity audio uniform backing `a.fft`.
pub const AUDIO_UNIFORM: &str = "_audio";

#[derive(Debug, PartialEq)]
pub struct ExprError(pub String);

impl std::fmt::Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Compile `() => …`'s body, or a bare expression, to GLSL.
pub fn compile(source: &str) -> Result<String, ExprError> {
    if source.len() > crate::program::MAX_HYDRA_SOURCE_BYTES {
        return Err(ExprError(format!(
            "a Hydra callback exceeds the {} byte limit",
            crate::program::MAX_HYDRA_SOURCE_BYTES
        )));
    }
    let body = source
        .trim()
        .strip_prefix("()")
        .and_then(|rest| rest.trim_start().strip_prefix("=>"))
        .unwrap_or(source);
    let mut reader = Expr {
        bytes: body.trim().as_bytes(),
        at: 0,
    };
    let glsl = reader.expression(0)?;
    reader.space();
    if reader.at < reader.bytes.len() {
        return Err(ExprError(format!(
            "`{}` is more than this compiles",
            String::from_utf8_lossy(reader.bytes)
        )));
    }
    Ok(glsl)
}

struct Expr<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Expr<'_> {
    fn space(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_whitespace())
        {
            self.at += 1;
        }
    }

    fn eat(&mut self, text: &str) -> bool {
        self.space();
        if self.bytes[self.at..].starts_with(text.as_bytes()) {
            self.at += text.len();
            return true;
        }
        false
    }

    fn expression(&mut self, depth: usize) -> Result<String, ExprError> {
        self.check_depth(depth)?;
        let mut left = self.term(depth)?;
        loop {
            self.space();
            let operator = match self.bytes.get(self.at) {
                Some(b'+') => "+",
                // `-` only binds here when something follows it as a term.
                Some(b'-') => "-",
                _ => return Ok(left),
            };
            self.at += 1;
            let right = self.term(depth)?;
            left = format!("({left} {operator} {right})");
        }
    }

    fn term(&mut self, depth: usize) -> Result<String, ExprError> {
        self.check_depth(depth)?;
        let mut left = self.unary(depth)?;
        loop {
            self.space();
            let operator = match self.bytes.get(self.at) {
                Some(b'*') => "*",
                Some(b'/') => "/",
                _ => return Ok(left),
            };
            self.at += 1;
            let right = self.unary(depth)?;
            left = format!("({left} {operator} {right})");
        }
    }

    fn unary(&mut self, depth: usize) -> Result<String, ExprError> {
        self.space();
        let mut negatives = 0usize;
        while self.eat("-") {
            negatives += 1;
            self.check_depth(depth.saturating_add(negatives))?;
        }
        let mut value = self.primary(depth.saturating_add(negatives))?;
        for _ in 0..negatives {
            value = format!("(-{value})");
        }
        Ok(value)
    }

    fn primary(&mut self, depth: usize) -> Result<String, ExprError> {
        self.check_depth(depth)?;
        self.space();
        if self.eat("(") {
            let inner = self.expression(depth + 1)?;
            if !self.eat(")") {
                return Err(ExprError("a bracket is never closed".into()));
            }
            return Ok(format!("({inner})"));
        }
        if self.eat("a.fft[") {
            let index = self.number()?;
            if !self.eat("]") {
                return Err(ExprError("`a.fft[` is never closed".into()));
            }
            let band = index
                .trim()
                .parse::<usize>()
                .unwrap_or(crate::HYDRA_AUDIO_BINS);
            if band >= crate::HYDRA_AUDIO_BINS {
                return Err(ExprError(format!("there is no audio band {band}")));
            }
            let vector = band / 4;
            let component = band % 4;
            return Ok(if vector == 0 {
                format!("{AUDIO_UNIFORM}[{component}]")
            } else {
                format!("{AUDIO_UNIFORM}{vector}[{component}]")
            });
        }
        if self.eat("Math.") {
            return self.call(depth);
        }
        if self.eat("time") {
            return Ok("time".into());
        }
        let number = self.number()?;
        // The digits go into the shader as written, and the shader reads them
        // as an f32. `1.2.3` would not lex, and a forty-digit integer - or
        // three hundred and ten of them, past even f64 - lexes as infinity;
        // either way composition would succeed and every pipeline build fail,
        // per frame and uncached, naming nothing. Refuse them here instead,
        // worded the way the chain parser words its own.
        let Ok(value) = number.parse::<f64>() else {
            return Err(ExprError(format!("`{number}` is not a number")));
        };
        if !crate::glsl::compose::shader_float(value) {
            return Err(ExprError(crate::glsl::compose::out_of_range(&number)));
        }
        // GLSL wants a float where JavaScript is happy with an integer.
        Ok(if number.contains('.') {
            number
        } else {
            format!("{number}.0")
        })
    }

    /// `Math.sin(x)`, and the handful of others a sketch reaches for.
    fn call(&mut self, depth: usize) -> Result<String, ExprError> {
        self.check_depth(depth)?;
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_alphabetic())
        {
            self.at += 1;
        }
        let name = String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned();
        // GLSL has all of these under the same names but `round`, which is
        // absent from the ES dialects hydra's own functions are written for.
        let glsl = match name.as_str() {
            "sin" | "cos" | "tan" | "floor" | "ceil" | "abs" | "sqrt" | "exp" | "log" | "sign" => {
                name.clone()
            }
            "round" => "ROUND".into(),
            "min" | "max" | "pow" | "atan" => name.clone(),
            "PI" => return Ok("3.1415926".into()),
            other => return Err(ExprError(format!("`Math.{other}` is not compiled"))),
        };
        if !self.eat("(") {
            return Err(ExprError(format!("`Math.{name}` needs a bracket")));
        }
        let mut args = vec![self.expression(depth + 1)?];
        while self.eat(",") {
            if args.len() >= crate::program::MAX_HYDRA_ARGS {
                return Err(ExprError(format!(
                    "a Hydra callback exceeds the {} argument limit",
                    crate::program::MAX_HYDRA_ARGS
                )));
            }
            args.push(self.expression(depth + 1)?);
        }
        if !self.eat(")") {
            return Err(ExprError(format!("`Math.{name}` is never closed")));
        }
        if glsl == "ROUND" {
            // `floor(x + 0.5)`, which is what `round` means and what every
            // GLSL version has.
            return Ok(format!("floor(({}) + 0.5)", args[0]));
        }
        Ok(format!("{glsl}({})", args.join(", ")))
    }

    fn check_depth(&self, depth: usize) -> Result<(), ExprError> {
        if depth > crate::program::MAX_HYDRA_DEPTH {
            return Err(ExprError(format!(
                "a Hydra callback exceeds the {} nesting-depth limit",
                crate::program::MAX_HYDRA_DEPTH
            )));
        }
        Ok(())
    }

    fn number(&mut self) -> Result<String, ExprError> {
        self.space();
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_digit() || *b == b'.')
        {
            self.at += 1;
        }
        if start == self.at {
            return Err(ExprError(format!(
                "`{}` is not something this compiles",
                String::from_utf8_lossy(&self.bytes[self.at..])
            )));
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_expressions_the_catalogue_uses_all_compile() {
        // Every distinct callback in the shipped snippets.
        for (source, expected) in [
            ("() => -0.18", "(-0.18)"),
            ("() => time * 0.05", "(time * 0.05)"),
            ("() => a.fft[0]", "_audio[0]"),
            ("() => a.fft[0] * 9", "(_audio[0] * 9.0)"),
            ("() => 0.8 + a.fft[0] * 1.4", "(0.8 + (_audio[0] * 1.4))"),
            (
                "() => 0.3 + Math.sin(time * 0.2)",
                "(0.3 + sin((time * 0.2)))",
            ),
            (
                "() => 3 + Math.floor(a.fft[1] * 6)",
                "(3.0 + floor((_audio[1] * 6.0)))",
            ),
        ] {
            assert_eq!(compile(source).as_deref(), Ok(expected), "{source}");
        }
    }

    #[test]
    fn what_it_cannot_compile_it_names() {
        assert!(compile("() => shape(4)").is_err(), "a chain is not a value");
        assert!(compile("() => Math.random()").is_err());
        assert_eq!(compile("() => a.fft[6]").as_deref(), Ok("_audio1[2]"));
        assert!(
            compile("() => a.fft[16]").is_err(),
            "there are sixteen bands"
        );
        assert!(compile("() => {fall}").is_err());
    }

    #[test]
    fn callback_size_and_recursive_shapes_are_bounded() {
        assert!(compile(&"-".repeat(crate::program::MAX_HYDRA_SOURCE_BYTES + 1)).is_err());
        assert!(
            compile(&format!(
                "{}0",
                "-".repeat(crate::program::MAX_HYDRA_DEPTH + 1)
            ))
            .is_err()
        );
        assert!(
            compile(&format!(
                "{}0{}",
                "(".repeat(crate::program::MAX_HYDRA_DEPTH + 1),
                ")".repeat(crate::program::MAX_HYDRA_DEPTH + 1)
            ))
            .is_err()
        );

        let boundary = format!(
            "{}0{}",
            "(".repeat(crate::program::MAX_HYDRA_DEPTH),
            ")".repeat(crate::program::MAX_HYDRA_DEPTH)
        );
        compile(&boundary).expect("the callback nesting boundary is accepted");
    }

    #[test]
    fn a_literal_no_shader_float_can_hold_is_refused_by_name() {
        // Digits only - this grammar has no exponent - so overflow is spelled
        // out: forty digits are finite as f64 but not as the f32 the shader
        // reads, and three hundred and ten overflow f64 itself.
        let forty = format!("1{}", "0".repeat(39));
        let past_f64 = format!("1{}", "0".repeat(309));
        // Just past the point where an f32 rounds to infinity.
        let past_max = "340282357000000000000000000000000000000";
        for literal in [forty.as_str(), past_f64.as_str(), past_max] {
            for source in [
                format!("() => {literal}"),
                format!("() => -{literal}"),
                format!("() => time * {literal}"),
                format!("() => Math.sin({literal}.5)"),
            ] {
                let error = compile(&source)
                    .expect_err("GLSL has no literal for an infinite float")
                    .to_string();
                assert!(
                    error.contains("is out of range for a shader float"),
                    "refused by name: `{source}` said {error}"
                );
                assert!(
                    error.contains(&format!("`{literal}")),
                    "names the literal: {error}"
                );
            }
        }

        // `f32::MAX` spelled out stays legal, and accepted literals keep the
        // exact text they always had.
        let max = "340282350000000000000000000000000000000";
        assert_eq!(compile(&format!("() => {max}")), Ok(format!("{max}.0")));
        assert_eq!(compile("() => 1.").as_deref(), Ok("1."));
        assert_eq!(compile("() => .5").as_deref(), Ok(".5"));
        assert_eq!(compile("() => 12").as_deref(), Ok("12.0"));

        // Digits and dots that are not a number at all would not lex either.
        for malformed in ["() => 1.2.3", "() => .", "() => 1..2"] {
            let error = compile(malformed).expect_err("not a number").to_string();
            assert!(error.contains("is not a number"), "{malformed}: {error}");
        }
    }
}
