//! Built-in bitmap font and its glyph atlas.
//!
//! The atlas is **static and complete**: every printable ASCII character is
//! rasterised once at construction into a small `R8_UNORM` image, so the
//! dynamic-atlas machinery ADR-0006 sketches (skyline packer, streamed
//! `glyph_blit.comp`, `fontdue`) has nothing left to do. It exists in the
//! ADR because glyphs arrive continuously as the user types *new*
//! characters; with all 95 of them resident from frame zero, they never do.
//! Re-introduce it when a real TTF and arbitrary code points land.
//!
//! Glyphs are authored as ASCII art — one line per character, rows separated
//! by `/` — because a hex table of 95 bit patterns is unreviewable and this
//! is not.
//!
//! Layout: a `GLYPH_W × GLYPH_H` design box, blitted into a `CELL_W × CELL_H`
//! atlas cell with a 1 px transparent margin on every side so bilinear taps
//! at a cell edge can never pull ink from the neighbouring glyph.

/// Design-box width of every glyph, in atlas texels.
pub const GLYPH_W: u32 = 5;
/// Design-box height. Rows 0..=6 are the cap band (baseline sits under row
/// 6); rows 7..=8 are the descender band.
pub const GLYPH_H: u32 = 9;
/// Horizontal advance per character, in design-box units — the glyph plus
/// one column of side bearing.
pub const ADVANCE: u32 = 6;

const CELL_W: u32 = GLYPH_W + 2;
const CELL_H: u32 = GLYPH_H + 2;
const COLS: u32 = 16;
// 16x6 exactly covers ASCII 32..=126 with one cell to spare; the seventh
// row is where the non-ASCII glyphs in `EXTRA` live.
const ROWS: u32 = 7;

/// First and last code points covered contiguously. Anything outside maps
/// to `?` unless it appears in [`EXTRA`].
const FIRST: u32 = 32;
const LAST: u32 = 126;

/// Glyphs past the ASCII block, appended to `GLYPHS` in this order.
///
/// Disclosure triangles: a tree row needs them, and `>` / `v` reads as text
/// rather than as a control. They are rotations of one another, which is why
/// the collapsed one is tall-and-narrow and the expanded one short-and-wide.
const EXTRA: [char; 2] = [ARROW_RIGHT, ARROW_DOWN];

/// Collapsed disclosure triangle.
pub const ARROW_RIGHT: char = '\u{25B8}';
/// Expanded disclosure triangle.
pub const ARROW_DOWN: char = '\u{25BE}';

pub const ATLAS_W: u32 = COLS * CELL_W;
pub const ATLAS_H: u32 = ROWS * CELL_H;

/// `'#'` = ink, `'.'` = transparent. Nine rows of five columns per glyph,
/// ASCII 32..=126 in order.
#[rustfmt::skip]
const GLYPHS: [&str; (LAST - FIRST + 1) as usize + EXTRA.len()] = [
    "...../...../...../...../...../...../...../...../.....", // ' '
    "..#../..#../..#../..#../..#../...../..#../...../.....", // '!'
    ".#.#./.#.#./...../...../...../...../...../...../.....", // '"'
    ".#.#./.#.#./#####/.#.#./#####/.#.#./.#.#./...../.....", // '#'
    "..#../.####/#.#../.###./..#.#/####./..#../...../.....", // '$'
    "##.../##..#/...#./..#../.#.../#..##/...##/...../.....", // '%'
    ".##../#..#./#.#../.#.../#.#.#/#..#./.##.#/...../.....", // '&'
    "..#../..#../...../...../...../...../...../...../.....", // '\''
    "...#./..#../.#.../.#.../.#.../..#../...#./...../.....", // '('
    ".#.../..#../...#./...#./...#./..#../.#.../...../.....", // ')'
    "...../#.#.#/.###./#####/.###./#.#.#/...../...../.....", // '*'
    "...../..#../..#../#####/..#../..#../...../...../.....", // '+'
    "...../...../...../...../...../..##./..#../.#.../.....", // ','
    "...../...../...../#####/...../...../...../...../.....", // '-'
    "...../...../...../...../...../..##./..##./...../.....", // '.'
    "....#/....#/...#./..#../.#.../#..../#..../...../.....", // '/'
    ".###./#...#/#..##/#.#.#/##..#/#...#/.###./...../.....", // '0'
    "..#../.##../..#../..#../..#../..#../.###./...../.....", // '1'
    ".###./#...#/....#/...#./..#../.#.../#####/...../.....", // '2'
    "#####/...#./..#../...#./....#/#...#/.###./...../.....", // '3'
    "...#./..##./.#.#./#..#./#####/...#./...#./...../.....", // '4'
    "#####/#..../####./....#/....#/#...#/.###./...../.....", // '5'
    "..##./.#.../#..../####./#...#/#...#/.###./...../.....", // '6'
    "#####/....#/...#./..#../.#.../.#.../.#.../...../.....", // '7'
    ".###./#...#/#...#/.###./#...#/#...#/.###./...../.....", // '8'
    ".###./#...#/#...#/.####/....#/...#./.##../...../.....", // '9'
    "...../..##./..##./...../..##./..##./...../...../.....", // ':'
    "...../..##./..##./...../..##./..#../.#.../...../.....", // ';'
    "...#./..#../.#.../#..../.#.../..#../...#./...../.....", // '<'
    "...../...../#####/...../#####/...../...../...../.....", // '='
    ".#.../..#../...#./....#/...#./..#../.#.../...../.....", // '>'
    ".###./#...#/....#/...#./..#../...../..#../...../.....", // '?'
    ".###./#...#/#.###/#.#.#/#.###/#..../.###./...../.....", // '@'
    "..#../.#.#./#...#/#...#/#####/#...#/#...#/...../.....", // 'A'
    "####./#...#/#...#/####./#...#/#...#/####./...../.....", // 'B'
    ".###./#...#/#..../#..../#..../#...#/.###./...../.....", // 'C'
    "###../#..#./#...#/#...#/#...#/#..#./###../...../.....", // 'D'
    "#####/#..../#..../####./#..../#..../#####/...../.....", // 'E'
    "#####/#..../#..../####./#..../#..../#..../...../.....", // 'F'
    ".###./#...#/#..../#.###/#...#/#...#/.####/...../.....", // 'G'
    "#...#/#...#/#...#/#####/#...#/#...#/#...#/...../.....", // 'H'
    ".###./..#../..#../..#../..#../..#../.###./...../.....", // 'I'
    "..###/...#./...#./...#./...#./#..#./.##../...../.....", // 'J'
    "#...#/#..#./#.#../##.../#.#../#..#./#...#/...../.....", // 'K'
    "#..../#..../#..../#..../#..../#..../#####/...../.....", // 'L'
    "#...#/##.##/#.#.#/#.#.#/#...#/#...#/#...#/...../.....", // 'M'
    "#...#/#...#/##..#/#.#.#/#..##/#...#/#...#/...../.....", // 'N'
    ".###./#...#/#...#/#...#/#...#/#...#/.###./...../.....", // 'O'
    "####./#...#/#...#/####./#..../#..../#..../...../.....", // 'P'
    ".###./#...#/#...#/#...#/#.#.#/#..#./.##.#/...../.....", // 'Q'
    "####./#...#/#...#/####./#.#../#..#./#...#/...../.....", // 'R'
    ".####/#..../#..../.###./....#/....#/####./...../.....", // 'S'
    "#####/..#../..#../..#../..#../..#../..#../...../.....", // 'T'
    "#...#/#...#/#...#/#...#/#...#/#...#/.###./...../.....", // 'U'
    "#...#/#...#/#...#/#...#/#...#/.#.#./..#../...../.....", // 'V'
    "#...#/#...#/#...#/#.#.#/#.#.#/##.##/#...#/...../.....", // 'W'
    "#...#/#...#/.#.#./..#../.#.#./#...#/#...#/...../.....", // 'X'
    "#...#/#...#/.#.#./..#../..#../..#../..#../...../.....", // 'Y'
    "#####/....#/...#./..#../.#.../#..../#####/...../.....", // 'Z'
    ".###./.#.../.#.../.#.../.#.../.#.../.###./...../.....", // '['
    "#..../#..../.#.../..#../...#./....#/....#/...../.....", // '\\'
    ".###./...#./...#./...#./...#./...#./.###./...../.....", // ']'
    "..#../.#.#./#...#/...../...../...../...../...../.....", // '^'
    "...../...../...../...../...../...../...../#####/.....", // '_'
    ".#.../..#../...../...../...../...../...../...../.....", // '`'
    "...../...../.###./....#/.####/#...#/.####/...../.....", // 'a'
    "#..../#..../####./#...#/#...#/#...#/####./...../.....", // 'b'
    "...../...../.###./#...#/#..../#...#/.###./...../.....", // 'c'
    "....#/....#/.####/#...#/#...#/#...#/.####/...../.....", // 'd'
    "...../...../.###./#...#/#####/#..../.###./...../.....", // 'e'
    "..##./.#..#/.#.../####./.#.../.#.../.#.../...../.....", // 'f'
    "...../...../.####/#...#/#...#/.####/....#/#...#/.###.", // 'g'
    "#..../#..../####./#...#/#...#/#...#/#...#/...../.....", // 'h'
    "..#../...../.##../..#../..#../..#../.###./...../.....", // 'i'
    "...#./...../..##./...#./...#./...#./...#./#..#./.##..", // 'j'
    "#..../#..../#..#./#.#../##.../#.#../#..#./...../.....", // 'k'
    ".##../..#../..#../..#../..#../..#../.###./...../.....", // 'l'
    "...../...../##.#./#.#.#/#.#.#/#.#.#/#...#/...../.....", // 'm'
    "...../...../####./#...#/#...#/#...#/#...#/...../.....", // 'n'
    "...../...../.###./#...#/#...#/#...#/.###./...../.....", // 'o'
    "...../...../####./#...#/#...#/####./#..../#..../#....", // 'p'
    "...../...../.####/#...#/#...#/.####/....#/....#/....#", // 'q'
    "...../...../#.##./##..#/#..../#..../#..../...../.....", // 'r'
    "...../...../.####/#..../.###./....#/####./...../.....", // 's'
    ".#.../.#.../####./.#.../.#.../.#..#/..##./...../.....", // 't'
    "...../...../#...#/#...#/#...#/#...#/.####/...../.....", // 'u'
    "...../...../#...#/#...#/#...#/.#.#./..#../...../.....", // 'v'
    "...../...../#...#/#.#.#/#.#.#/#.#.#/.#.#./...../.....", // 'w'
    "...../...../#...#/.#.#./..#../.#.#./#...#/...../.....", // 'x'
    "...../...../#...#/#...#/#...#/.####/....#/#...#/.###.", // 'y'
    "...../...../#####/...#./..#../.#.../#####/...../.....", // 'z'
    "...##/..#../..#../.##../..#../..#../...##/...../.....", // '{'
    "..#../..#../..#../..#../..#../..#../..#../...../.....", // '|'
    "##.../..#../..#../..##./..#../..#../##.../...../.....", // '}'
    ".##.#/#..#./...../...../...../...../...../...../.....", // '~'
    "...../...../.#.../.##../.###./.##../.#.../...../.....", // ARROW_RIGHT
    "...../...../...../#####/.###./..#../...../...../.....", // ARROW_DOWN
];

/// Cell index for a character; unmapped code points render as `?`.
fn cell(ch: char) -> u32 {
    let c = ch as u32;
    if (FIRST..=LAST).contains(&c) {
        return c - FIRST;
    }
    match EXTRA.iter().position(|&e| e == ch) {
        Some(i) => LAST - FIRST + 1 + i as u32,
        None => '?' as u32 - FIRST,
    }
}

/// Normalised `(u0, v0, u1, v1)` of a character's design box in the atlas.
pub fn glyph_uv(ch: char) -> [f32; 4] {
    let i = cell(ch);
    let x = (i % COLS) * CELL_W + 1;
    let y = (i / COLS) * CELL_H + 1;
    [
        x as f32 / ATLAS_W as f32,
        y as f32 / ATLAS_H as f32,
        (x + GLYPH_W) as f32 / ATLAS_W as f32,
        (y + GLYPH_H) as f32 / ATLAS_H as f32,
    ]
}

/// Rasterise every glyph into one `R8_UNORM` image of `ATLAS_W × ATLAS_H`
/// texels. Runs once, at `UiGpu` construction.
pub fn rasterize_atlas() -> Vec<u8> {
    let mut px = vec![0u8; (ATLAS_W * ATLAS_H) as usize];
    for (i, art) in GLYPHS.iter().enumerate() {
        let ox = (i as u32 % COLS) * CELL_W + 1;
        let oy = (i as u32 / COLS) * CELL_H + 1;
        for (row, line) in art.split('/').enumerate() {
            assert!(
                line.len() == GLYPH_W as usize && row < GLYPH_H as usize,
                "glyph {i} is not {GLYPH_W}x{GLYPH_H}",
            );
            for (col, b) in line.bytes().enumerate() {
                if b == b'#' {
                    px[((oy + row as u32) * ATLAS_W + ox + col as u32) as usize] = 255;
                }
            }
        }
    }
    px
}

/// Width in design-box units of a string laid out at scale 1. The trailing
/// side bearing of the last character is trimmed.
pub fn text_width(s: &str) -> u32 {
    match s.chars().count() as u32 {
        0 => 0,
        n => n * ADVANCE - (ADVANCE - GLYPH_W),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both disclosure triangles must land in the atlas with ink in them —
    /// a glyph that maps to a valid cell but rasterises blank is invisible
    /// with no other symptom.
    #[test]
    fn disclosure_glyphs_are_rasterised() {
        let px = rasterize_atlas();
        for ch in EXTRA {
            let i = cell(ch);
            let (ox, oy) = ((i % COLS) * CELL_W + 1, (i / COLS) * CELL_H + 1);
            let ink = (0..GLYPH_H)
                .flat_map(|r| (0..GLYPH_W).map(move |c| (r, c)))
                .filter(|(r, c)| px[((oy + r) * ATLAS_W + ox + c) as usize] != 0)
                .count();
            assert!(ink > 0, "{ch:?} (cell {i}) rasterised blank");
        }
        assert_ne!(cell(ARROW_RIGHT), cell(ARROW_DOWN));
        assert_ne!(cell(ARROW_RIGHT), cell('?'), "must not fall back to '?'");
    }
}
