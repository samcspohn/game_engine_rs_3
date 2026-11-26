//! Entity names: interned once per process, unique among siblings.
//!
//! See [names](../../../../docs/notes/names.md).

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::num::NonZeroU32;
use std::sync::LazyLock;

/// A string interned for the life of the process.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(NonZeroU32);

#[derive(Default)]
struct Interner {
    ids: FxMap<&'static str, Symbol>,
    strs: Vec<&'static str>,
}

static INTERNER: LazyLock<RwLock<Interner>> = LazyLock::new(Default::default);

/// Set in a symbol whose string ends in a digit, so suffixing needs no lock.
const DIGIT: u32 = 1 << 31;

impl Symbol {
    pub fn new(s: &str) -> Symbol {
        if let Some(&sym) = INTERNER.read().ids.get(s) {
            return sym;
        }
        let mut i = INTERNER.write();
        if let Some(&sym) = i.ids.get(s) {
            return sym;
        }
        let s: &'static str = Box::leak(s.into());
        let digit = if s.ends_with(|c: char| c.is_ascii_digit()) {
            DIGIT
        } else {
            0
        };
        let sym =
            Symbol(NonZeroU32::new((i.strs.len() as u32 + 1) | digit).expect("under 2G names"));
        i.strs.push(s);
        i.ids.insert(s, sym);
        sym
    }

    /// The symbol for `s` if anything has interned it, without interning it.
    pub fn find(s: &str) -> Option<Symbol> {
        INTERNER.read().ids.get(s).copied()
    }

    pub fn as_str(self) -> &'static str {
        INTERNER.read().strs[(self.0.get() & !DIGIT) as usize - 1]
    }

    fn ends_in_digit(self) -> bool {
        self.0.get() & DIGIT != 0
    }
}

/// A name, as a base and a numeric suffix: `ship12` is `ship` and 12. Copy,
/// and allocation-free once its base is interned. The default is anonymous.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Name {
    base: Option<Symbol>,
    n: u32,
}

impl Name {
    pub const NONE: Name = Name { base: None, n: 0 };

    pub fn new(s: &str) -> Name {
        let (base, n) = split(s);
        Name {
            base: (!base.is_empty()).then(|| Symbol::new(base)),
            n,
        }
    }

    /// The name `s` spells if any name could, without interning it.
    pub fn find(s: &str) -> Option<Name> {
        Name::find_in(&INTERNER.read(), s)
    }

    fn find_in(i: &Interner, s: &str) -> Option<Name> {
        let (base, n) = split(s);
        match base.is_empty() {
            true => Some(Name::NONE),
            false => Some(Name {
                base: Some(*i.ids.get(base)?),
                n,
            }),
        }
    }

    pub fn is_none(self) -> bool {
        self.base.is_none()
    }
}

/// `ship12` into `ship` and 12. Digits that a suffix would not print back
/// the same way (`ship012`, `12`, past `u32`) stay in the base.
fn split(s: &str) -> (&str, u32) {
    let base = s.trim_end_matches(|c: char| c.is_ascii_digit());
    let digits = &s[base.len()..];
    match (base.is_empty() || digits.starts_with('0'), digits.parse()) {
        (false, Ok(n)) => (base, n),
        _ => (s, 0),
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let base = self.base.map_or("", Symbol::as_str);
        match self.n {
            0 => f.write_str(base),
            n => write!(f, "{base}{n}"),
        }
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.to_string())
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Name {
        Name::new(s)
    }
}

impl From<&String> for Name {
    fn from(s: &String) -> Name {
        Name::new(s)
    }
}

impl From<String> for Name {
    fn from(s: String) -> Name {
        Name::new(&s)
    }
}

impl PartialEq<str> for Name {
    fn eq(&self, s: &str) -> bool {
        Name::find(s) == Some(*self)
    }
}

impl PartialEq<&str> for Name {
    fn eq(&self, s: &&str) -> bool {
        *self == **s
    }
}

/// Multiply-rotate hashing, for keys that are already small integers.
#[derive(Default)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, mut bytes: &[u8]) {
        while let Some((word, rest)) = bytes.split_first_chunk() {
            self.write_u64(u64::from_le_bytes(*word));
            bytes = rest;
        }
        if !bytes.is_empty() {
            self.write_u64(bytes.iter().rev().fold(0, |a, &b| a << 8 | b as u64));
        }
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(n as u64);
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    // The product's well-mixed high bits down to where the table indexes.
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// Every named slot by its parent and name, which is what keeps siblings'
/// names unique and a lookup one probe per path segment.
#[derive(Default)]
pub(crate) struct Names {
    by: FxMap<(u32, Name), u32>,
    /// The next suffix to try for a name that was taken again. Released with
    /// the name, so nothing outlives a parent. Apart from `by`, which stays
    /// 16 bytes an entry: 20 made a lookup 15 ns slower.
    next: FxMap<(u32, Name), u32>,
}

impl Names {
    /// Record `idx` under `parent` as `name`, or as the next `name1`, `name2`
    /// free there. Returns the name it got.
    pub(crate) fn claim(&mut self, parent: u32, name: Name, idx: u32) -> Name {
        let Some(base) = name.base else {
            return name;
        };
        if !self.by.contains_key(&(parent, name)) {
            self.by.insert((parent, name), idx);
            return name;
        }
        // `x0` suffixed would read back as `x01`, which is not 1.
        let suffixed = match base.ends_in_digit() {
            true => Symbol::new(&format!("{}_", base.as_str())),
            false => base,
        };
        let next = self.next.entry((parent, name)).or_default();
        let mut k = (*next).max(name.n + 1);
        let got = loop {
            let n = Name {
                base: Some(suffixed),
                n: k,
            };
            k += 1;
            if !self.by.contains_key(&(parent, n)) {
                break n;
            }
        };
        *next = k;
        self.by.insert((parent, got), idx);
        got
    }

    pub(crate) fn release(&mut self, parent: u32, name: Name) {
        if !name.is_none() {
            self.by.remove(&(parent, name));
            if !self.next.is_empty() {
                self.next.remove(&(parent, name));
            }
        }
    }

    pub(crate) fn child(&self, parent: u32, name: Name) -> Option<u32> {
        self.by.get(&(parent, name)).copied()
    }
}

/// A path parsed once, for one looked up often.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Path {
    absolute: bool,
    names: Box<[Name]>,
}

impl Path {
    pub fn new(path: &str) -> Path {
        Path {
            absolute: path.starts_with('/'),
            names: segments(path).map(Name::new).collect(),
        }
    }

    pub(super) fn relative(names: Vec<Name>) -> Path {
        Path {
            absolute: false,
            names: names.into(),
        }
    }
}

fn segments(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').filter(|s| !s.is_empty())
}

/// What [`TransformHierarchy::find`](super::TransformHierarchy::find) takes:
/// a `&str` or a [`Path`]. One starting with `/` starts at the root, and its
/// first name is the root's own.
pub trait Lookup {
    /// Whether it is absolute, and its names: `None` for one nothing has.
    fn parts(&self) -> (bool, impl Iterator<Item = Option<Name>>);
}

impl Lookup for str {
    fn parts(&self) -> (bool, impl Iterator<Item = Option<Name>>) {
        let i = INTERNER.read();
        (
            self.starts_with('/'),
            segments(self).map(move |s| Name::find_in(&i, s)),
        )
    }
}

impl Lookup for Path {
    fn parts(&self) -> (bool, impl Iterator<Item = Option<Name>>) {
        (self.absolute, self.names.iter().map(|&n| Some(n)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_prints_back_as_it_was_spelled() {
        for s in [
            "ship",
            "ship12",
            "ship012",
            "12",
            "ship0",
            "a:prim3",
            "",
            "x99999999999",
        ] {
            assert_eq!(Name::new(s).to_string(), s);
            assert_eq!(Name::new(s), *s);
        }
    }

    #[test]
    fn a_taken_name_gets_the_next_free_suffix() {
        let mut n = Names::default();
        let got: Vec<String> = ["ship", "ship", "ship2", "ship", "v2", "v2", "x0", "x0"]
            .iter()
            .enumerate()
            .map(|(i, s)| n.claim(7, Name::new(s), i as u32).to_string())
            .collect();
        assert_eq!(
            got,
            ["ship", "ship1", "ship2", "ship3", "v2", "v3", "x0", "x0_1"]
        );
        assert_eq!(n.claim(8, Name::new("ship"), 9).to_string(), "ship");
        assert!(n.claim(7, Name::NONE, 10).is_none());
        assert!(n.claim(7, Name::NONE, 11).is_none());
    }

    #[test]
    fn a_name_retaken_after_its_holder_went_steps_past_the_suffixes_left() {
        let mut n = Names::default();
        let ship = Name::new("ship");
        let got: Vec<String> = (0..3).map(|i| n.claim(7, ship, i).to_string()).collect();
        assert_eq!(got, ["ship", "ship1", "ship2"]);
        n.release(7, ship);
        assert_eq!(n.claim(7, ship, 3).to_string(), "ship");
        assert_eq!(n.claim(7, ship, 4).to_string(), "ship3");
    }
}
