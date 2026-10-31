//! Compile the small expressions a Hydra sketch writes as callbacks.
//!
//! Callback expressions such as `osc(() => time * 0.05)` are compiled into
//! GLSL. The shader reads `time`, the pointer and audio bands from uniforms;
//! the renderer does not execute JavaScript callbacks.
//!
//! Supported expressions are limited to numbers, parentheses, JavaScript's
//! arithmetic, comparison and logic operators, `? :`, the `Math` functions
//! and constants in [`Expr::call`], `time`, `width`, `height`, `mouse.x`,
//! `mouse.y` and `a.fft[n]`. Every value is a float, as JavaScript would
//! give it in arithmetic: a comparison is 1 or 0, and 0 is false.
//! Unsupported expressions are rejected rather than silently changing the
//! sketch's behavior. `Math.random()` is a new value each frame, the same on
//! every pixel, as a callback Hydra runs once a frame gives. It hashes
//! `time`, so it holds while the clock stands still.

/// The first `vec4` in the fixed-capacity audio uniform backing `a.fft`.
pub const AUDIO_UNIFORM: &str = "_audio";

/// The pointer behind `mouse.x` and `mouse.y`, in pixels of the picture from
/// its top-left corner, as hydra's `mouse` reads the page.
pub const MOUSE_UNIFORM: &str = "_mouse";

/// Binary operators from the loosest to the tightest, as JavaScript ranks
/// them. Within a level a longer operator comes first.
const LEVELS: &[&[&str]] = &[
    &["||"],
    &["&&"],
    &["===", "!==", "==", "!="],
    &["<=", ">=", "<", ">"],
    &["+", "-"],
    &["*", "/", "%"],
];

/// The functions a compiled callback can call, by the text that calls them.
/// A shader carries one only when its body calls it. Each keeps an operand
/// from being written twice, which nesting would multiply.
pub const HELPERS: &[(&str, &str)] = &[
    // JavaScript's `%` keeps the sign of the left side, where GLSL's `mod`
    // keeps the sign of the right.
    (
        "_remainder(",
        "float _remainder(float a, float b) { float q = a / b; return a - b * (q < 0.0 ? ceil(q) : floor(q)); }",
    ),
    (
        "_trunc(",
        "float _trunc(float x) { return x < 0.0 ? ceil(x) : floor(x); }",
    ),
    // `&&` and `||` give one of their sides, as in JavaScript.
    (
        "_and(",
        "float _and(float a, float b) { return a != 0.0 ? b : a; }",
    ),
    (
        "_or(",
        "float _or(float a, float b) { return a != 0.0 ? a : b; }",
    ),
];

fn binary_glsl(operator: &str, left: &str, right: &str) -> String {
    match operator {
        "%" => format!("_remainder({left}, {right})"),
        "&&" => format!("_and({left}, {right})"),
        "||" => format!("_or({left}, {right})"),
        "===" | "==" => format!("float({left} == {right})"),
        "!==" | "!=" => format!("float({left} != {right})"),
        "<=" | ">=" | "<" | ">" => format!("float({left} {operator} {right})"),
        _ => format!("({left} {operator} {right})"),
    }
}

#[derive(Debug, PartialEq)]
pub struct ExprError(pub String);

impl std::fmt::Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Compile `() => …`'s body, or a bare expression, to GLSL.
pub fn compile(source: &str) -> Result<String, ExprError> {
    compile_seeded(source, &mut 0)
}

/// The same, numbering each `Math.random()` from `randoms`, so every call in
/// one shader has a value of its own.
pub fn compile_seeded(source: &str, randoms: &mut u32) -> Result<String, ExprError> {
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
        randoms,
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
    /// `Math.random()` calls so far in the shader, so each has a value of
    /// its own.
    randoms: &'a mut u32,
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

    /// `test ? yes : no`, the loosest form, then the binary operators.
    fn expression(&mut self, depth: usize) -> Result<String, ExprError> {
        self.check_depth(depth)?;
        let test = self.binary(0, depth)?;
        if !self.eat("?") {
            return Ok(test);
        }
        let yes = self.expression(depth + 1)?;
        if !self.eat(":") {
            return Err(ExprError("a `?` has no `:`".into()));
        }
        let no = self.expression(depth + 1)?;
        Ok(format!("({test} != 0.0 ? {yes} : {no})"))
    }

    /// One level of [`LEVELS`], left to right, each side a tighter level.
    fn binary(&mut self, level: usize, depth: usize) -> Result<String, ExprError> {
        let Some(operators) = LEVELS.get(level) else {
            return self.unary(depth);
        };
        self.check_depth(depth)?;
        let mut left = self.binary(level + 1, depth)?;
        while let Some(operator) = self.operator(operators) {
            let right = self.binary(level + 1, depth)?;
            left = binary_glsl(operator, &left, &right);
        }
        Ok(left)
    }

    /// The first of `operators` the text goes on with. A level lists a
    /// longer operator first, so `<=` is never read as `<`.
    fn operator(&mut self, operators: &[&'static str]) -> Option<&'static str> {
        self.space();
        let rest = &self.bytes[self.at..];
        let operator = operators
            .iter()
            .copied()
            .find(|operator| rest.starts_with(operator.as_bytes()))?;
        self.at += operator.len();
        Some(operator)
    }

    fn unary(&mut self, depth: usize) -> Result<String, ExprError> {
        self.space();
        let mut prefixes = Vec::new();
        loop {
            let prefix = if self.eat("-") {
                '-'
            } else if self.eat("!") {
                '!'
            } else if self.eat("+") {
                '+'
            } else {
                break;
            };
            prefixes.push(prefix);
            self.check_depth(depth.saturating_add(prefixes.len()))?;
        }
        let mut value = self.primary(depth.saturating_add(prefixes.len()))?;
        for prefix in prefixes.iter().rev() {
            value = match prefix {
                '-' => format!("(-{value})"),
                '!' => format!("float({value} == 0.0)"),
                _ => value,
            };
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
        if self.eat("width") {
            return Ok("resolution.x".into());
        }
        if self.eat("height") {
            return Ok("resolution.y".into());
        }
        if self.eat("mouse.x") {
            return Ok(format!("{MOUSE_UNIFORM}.x"));
        }
        if self.eat("mouse.y") {
            return Ok(format!("{MOUSE_UNIFORM}.y"));
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

    /// `Math.sin(x)`, the other functions a sketch reaches for and the
    /// constants, each function taking as many values as JavaScript reads.
    fn call(&mut self, depth: usize) -> Result<String, ExprError> {
        self.check_depth(depth)?;
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            self.at += 1;
        }
        let name = String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned();
        // To the digits an f32 holds.
        let constant = match name.as_str() {
            "PI" => Some("3.1415926"),
            "E" => Some("2.7182818"),
            "LN2" => Some("0.6931472"),
            "LN10" => Some("2.3025851"),
            "LOG2E" => Some("1.4426950"),
            "LOG10E" => Some("0.4342945"),
            "SQRT2" => Some("1.4142136"),
            "SQRT1_2" => Some("0.7071068"),
            _ => None,
        };
        if let Some(constant) = constant {
            return Ok(constant.into());
        }
        if name == "random" {
            if !(self.eat("(") && self.eat(")")) {
                return Err(ExprError("`Math.random` takes no arguments".into()));
            }
            // A hash of the frame's time: one value for every pixel, a new
            // one each frame. Each call in the shader has its own.
            *self.randoms += 1;
            return Ok(format!(
                "fract(sin(time * 78.233 + {}.0 * 12.9898) * 43758.5453)",
                *self.randoms
            ));
        }
        let (least, most) = match name.as_str() {
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "floor" | "ceil" | "round"
            | "trunc" | "abs" | "sqrt" | "exp" | "log" | "log2" | "log10" | "sign" => (1, 1),
            "pow" | "atan2" => (2, 2),
            "hypot" => (1, 4),
            "min" | "max" => (1, crate::program::MAX_HYDRA_ARGS),
            other => return Err(ExprError(format!("`Math.{other}` is not compiled"))),
        };
        if !self.eat("(") {
            return Err(ExprError(format!("`Math.{name}` needs a bracket")));
        }
        let mut args = Vec::new();
        if !self.eat(")") {
            args.push(self.expression(depth + 1)?);
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
        }
        if !(least..=most).contains(&args.len()) {
            let count = if least == most {
                least.to_string()
            } else {
                format!("{least} to {most}")
            };
            let unit = if most == 1 { "value" } else { "values" };
            return Err(ExprError(format!(
                "`Math.{name}` takes {count} {unit} here"
            )));
        }
        // GLSL has the rest under the same names. `round` is absent from the
        // ES dialects hydra's own functions are written for, and `floor(x +
        // 0.5)` is what JavaScript's `round` means.
        Ok(match name.as_str() {
            "round" => format!("floor(({}) + 0.5)", args[0]),
            "trunc" => format!("_trunc({})", args[0]),
            "log10" => format!("(log({}) * 0.4342945)", args[0]),
            "atan2" => format!("atan({}, {})", args[0], args[1]),
            "hypot" if args.len() == 1 => format!("abs({})", args[0]),
            "hypot" => format!("length(vec{}({}))", args.len(), args.join(", ")),
            "min" | "max" => args
                .into_iter()
                .reduce(|left, right| format!("{name}({left}, {right})"))
                .unwrap_or_default(),
            _ => format!("{name}({})", args.join(", ")),
        })
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
        assert!(compile("() => Math.random(1)").is_err());
        assert_eq!(compile("() => a.fft[6]").as_deref(), Ok("_audio1[2]"));
        assert!(
            compile("() => a.fft[16]").is_err(),
            "there are sixteen bands"
        );
        assert!(compile("() => {fall}").is_err());
    }

    #[test]
    fn the_size_and_a_random_value_read_the_frame() {
        assert_eq!(
            compile("() => width / height").as_deref(),
            Ok("(resolution.x / resolution.y)")
        );
        let twice = compile("() => Math.random() - Math.random()").expect("random");
        assert!(
            twice.contains("+ 1.0 * 12.9898") && twice.contains("+ 2.0 * 12.9898"),
            "each call has its own value: {twice}"
        );
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

    #[test]
    fn the_pointer_reads_its_uniform() {
        assert_eq!(
            compile("() => mouse.x / width - mouse.y").as_deref(),
            Ok("((_mouse.x / resolution.x) - _mouse.y)")
        );
        assert!(
            compile("() => mouse.z").is_err(),
            "the pointer has two axes"
        );
        assert!(compile("() => mouse").is_err(), "a value, not the object");
    }

    #[test]
    fn comparisons_and_conditions_are_floats_as_javascript_counts_them() {
        for (source, expected) in [
            ("() => time < 2", "float(time < 2.0)"),
            ("() => time >= 2 === 1", "float(float(time >= 2.0) == 1.0)"),
            ("() => !time", "float(time == 0.0)"),
            ("() => -!time", "(-float(time == 0.0))"),
            ("() => +time", "time"),
            (
                "() => time > 1 ? 2 : 3",
                "(float(time > 1.0) != 0.0 ? 2.0 : 3.0)",
            ),
            (
                "() => time ? 1 : time ? 2 : 3",
                "(time != 0.0 ? 1.0 : (time != 0.0 ? 2.0 : 3.0))",
            ),
            ("() => time % 4", "_remainder(time, 4.0)"),
            (
                "() => 1 + time % 4 * 2",
                "(1.0 + (_remainder(time, 4.0) * 2.0))",
            ),
            ("() => time && 1 || 2", "_or(_and(time, 1.0), 2.0)"),
            (
                "() => 1 + 2 < 4 == 1",
                "float(float((1.0 + 2.0) < 4.0) == 1.0)",
            ),
        ] {
            assert_eq!(compile(source).as_deref(), Ok(expected), "{source}");
        }
        for broken in [
            "() => time ? 1",
            "() => time = 1",
            "() => time ** 2",
            "() => time & 1",
            "() => time ?? 1",
        ] {
            assert!(compile(broken).is_err(), "{broken} is refused");
        }
    }

    #[test]
    fn math_reads_as_many_values_as_javascript_does() {
        for (source, expected) in [
            ("() => Math.atan2(1, 2)", "atan(1.0, 2.0)"),
            ("() => Math.hypot(3)", "abs(3.0)"),
            ("() => Math.hypot(3, 4)", "length(vec2(3.0, 4.0))"),
            ("() => Math.min(1, 2, 3)", "min(min(1.0, 2.0), 3.0)"),
            ("() => Math.max(time)", "time"),
            ("() => Math.trunc(time)", "_trunc(time)"),
            ("() => Math.log10(time)", "(log(time) * 0.4342945)"),
            ("() => Math.E * Math.SQRT1_2", "(2.7182818 * 0.7071068)"),
        ] {
            assert_eq!(compile(source).as_deref(), Ok(expected), "{source}");
        }
        for (source, said) in [
            ("() => Math.sin(1, 2)", "`Math.sin` takes 1 value here"),
            ("() => Math.pow(2)", "`Math.pow` takes 2 values here"),
            (
                "() => Math.hypot()",
                "`Math.hypot` takes 1 to 4 values here",
            ),
            ("() => Math.cbrt(8)", "`Math.cbrt` is not compiled"),
        ] {
            assert_eq!(compile(source), Err(ExprError(said.into())), "{source}");
        }
    }
}
