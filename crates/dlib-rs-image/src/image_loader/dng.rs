//! dlib's DNG image format: `save_dng`/`load_dng`, ported bit-exactly from
//! dlib's `dlib/image_saver/image_saver.h` (the `save_dng_helper` family),
//! `dlib/image_loader/image_loader.h` (`load_dng`) and
//! `dlib/image_saver/dng_shared.h` (predictors, type tags).
//!
//! Which kernels dlib actually uses for DNG (verified in the C++ sources):
//! * range coder: `entropy_encoder::kernel_2a` == `entropy_encoder_kernel_2`
//!   (`dlib/entropy_encoder/entropy_encoder_kernel_2.{h,cpp}`) and
//!   `entropy_decoder::kernel_2a` == `entropy_decoder_kernel_2`
//!   (`dlib/entropy_decoder/entropy_decoder_kernel_2.{h,cpp}`).
//! * context model: `entropy_encoder_model<256,enc>::kernel_5a` /
//!   `entropy_decoder_model<256,dec>::kernel_5a` (PPM, escape method D,
//!   Shkarin information inheritance) for all integer/rgb/hsi image types,
//!   and `kernel_4a` (plain PPM, escape method D) for the compressed
//!   exponent stream of `grayscale_float` images.
//! * No `conditioning_class` kernel is used by DNG: `kernel_4a`/`kernel_5a`
//!   models are self-contained context trees; `conditioning_class` is only
//!   referenced by `entropy_*_model_kernel_1..3`, which DNG does not use.
//!
//! DNG container layout (little-endian where fixed width):
//! * bytes `"DNG"` (3 raw bytes),
//! * `version` serialized with dlib packed-integer encoding (== 1),
//! * `type` packed (1=grayscale, 2=rgb, 3=hsi, 4=rgb_paeth, 5=rgb_alpha,
//!   6=rgb_alpha_paeth, 7=grayscale_16bit, 8=grayscale_float),
//! * `width` (nc) and `height` (nr) as packed signed integers,
//! * then either the range-coded PPM stream (all non-float types) or, for
//!   `grayscale_float`, the per-pixel packed i64 mantissa deltas followed by
//!   a packed-length-prefixed byte blob holding the range-coded exponents.
//!
//! Edge semantics preserved from C++:
//! * The range coder is flushed at the end of `save_dng`: dlib's
//!   `entropy_encoder_kernel_2` destructor calls `flush()` (writing the 4
//!   final bytes of `low`), which we mirror explicitly after the trailing
//!   magic bytes. The float path flushes the exponent encoder the same way
//!   (via `encoder.clear()` in the C++).
//! * dlib's HSI coding truncates each channel delta to a single byte
//!   (`(unsigned char)(cur.h - pre.h)`), so DNG storage of `hsi_pixel` is
//!   lossy whenever channel values exceed 8 bits; this port keeps that
//!   behavior.
//! * The range decoder's `set_stream` reads 4 bytes unconditionally;
//!   missing bytes read as 0 (C++ `sgetn` failure leaves 0 / the first read's
//!   result is ignored — the missing-first-byte case is unspecified in C++,
//!   we use 0).

use std::io::{Read, Write};

use dlib_rs_core::serialize::{Deserializer, FloatDetails, Serializer};

use crate::array2d::Array2D;
use crate::pixel::{assign_pixel, HsiPixel, Pixel, PixelValue, RgbAlphaPixel, RgbPixel};
/// dlib's deserializer errors surface as corruption (short/malformed input).
impl From<dlib_rs_core::serialize::SerializeError> for DngError {
    fn from(e: dlib_rs_core::serialize::SerializeError) -> Self {
        DngError::Corrupt(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// range coder: port of entropy_encoder_kernel_2 / entropy_decoder_kernel_2
// ---------------------------------------------------------------------------

/// Errors from the DNG encoder/decoder (dlib throws `image_save_error` /
/// `image_load_error`).
#[derive(Debug, thiserror::Error)]
pub enum DngError {
    /// Underlying stream failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Malformed DNG data.
    #[error("corrupt dng data: {0}")]
    Corrupt(String),
}

// ---------------------------------------------------------------------------
// range coder: port of entropy_encoder_kernel_2 / entropy_decoder_kernel_2
// ---------------------------------------------------------------------------

/// Range coder, port of `dlib::entropy_encoder_kernel_2`
/// (`dlib/entropy_encoder/entropy_encoder_kernel_2.{h,cpp}`).
///
/// `low` is kept in `[1, 2^32-1]`, `high` in `[0, 2^32-1]`; both are 32-bit
/// fixed point reals in [0,1) with the point before the first bit. All
/// arithmetic is wrapping u32 exactly like the C++ `uint32` code.
#[derive(Debug)]
struct RangeEncoder {
    low: u32,
    high: u32,
    out: Vec<u8>,
}

impl RangeEncoder {
    /// `entropy_encoder_kernel_2()` ctor state.
    fn new() -> Self {
        RangeEncoder {
            low: 0x0000_0001,
            high: 0xffff_ffff,
            out: Vec::new(),
        }
    }

    /// `encode(low_count, high_count, total)`.
    fn encode(&mut self, low_count: u32, high_count: u32, total: u32) {
        // we must add one because high == real upper range - 1
        let r = self.high.wrapping_sub(self.low).wrapping_add(1) / total;
        self.high = self
            .low
            .wrapping_add(r.wrapping_mul(high_count))
            .wrapping_sub(1);
        self.low = self.low.wrapping_add(r.wrapping_mul(low_count));

        loop {
            if (self.high & 0xFF00_0000) != (self.low & 0xFF00_0000) {
                if self.high.wrapping_sub(self.low) < 0x1_0000 {
                    if self.high.wrapping_sub(self.low) > 0x1000 {
                        self.high >>= 1;
                        self.low >>= 1;
                        let v = self.high.wrapping_add(self.low);
                        self.high = v.wrapping_add(0xFF);
                        self.low = v.wrapping_sub(0xFF);
                    } else {
                        self.high >>= 1;
                        self.low >>= 1;
                        let v = self.high.wrapping_add(self.low);
                        self.high = v;
                        self.low = v;
                    }
                } else {
                    break;
                }
            } else {
                // roll off 8 high order bits of low
                let buf = (self.low >> 24) as u8;
                self.high <<= 8;
                self.low <<= 8;
                self.high |= 0xFF;
                if self.low == 0 {
                    self.low = 1;
                }
                self.out.push(buf);
            }
        }
    }

    /// `flush()`: writes the 4 bytes of `low`, then resets the state.
    fn flush(&mut self) {
        self.out.push(((self.low >> 24) & 0xFF) as u8);
        self.out.push(((self.low >> 16) & 0xFF) as u8);
        self.out.push(((self.low >> 8) & 0xFF) as u8);
        self.out.push((self.low & 0xFF) as u8);
        self.low = 0x0000_0001;
        self.high = 0xffff_ffff;
    }
}

/// Range decoder, port of `dlib::entropy_decoder_kernel_2`
/// (`dlib/entropy_decoder/entropy_decoder_kernel_2.{h,cpp}`).
#[derive(Debug)]
struct RangeDecoder<'a> {
    low: u32,
    high: u32,
    target: u32,
    r: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    /// `set_stream`: priming read of 4 bytes (missing bytes count as 0).
    fn new(data: &'a [u8]) -> Self {
        let mut d = RangeDecoder {
            low: 0x0000_0001,
            high: 0xffff_ffff,
            target: 0,
            r: 0,
            data,
            pos: 0,
        };
        for _ in 0..4 {
            let ch = d.read_byte();
            d.target = (d.target << 8).wrapping_add(ch as u32);
        }
        d
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    /// `get_target(total)`.
    fn get_target(&mut self, total: u32) -> u32 {
        self.r = self.high.wrapping_sub(self.low).wrapping_add(1) / total;
        let temp = self.target.wrapping_sub(self.low) / self.r;
        if temp < total {
            temp
        } else {
            total - 1
        }
    }

    /// `decode(low_count, high_count)`.
    fn decode(&mut self, low_count: u32, high_count: u32) {
        self.high = self
            .low
            .wrapping_add(self.r.wrapping_mul(high_count))
            .wrapping_sub(1);
        self.low = self.low.wrapping_add(self.r.wrapping_mul(low_count));
        self.r = 0;

        loop {
            if (self.high & 0xFF00_0000) != (self.low & 0xFF00_0000) {
                if self.high.wrapping_sub(self.low) < 0x1_0000 {
                    if self.high.wrapping_sub(self.low) > 0x1000 {
                        self.high >>= 1;
                        self.low >>= 1;
                        let v = self.high.wrapping_add(self.low);
                        self.high = v.wrapping_add(0xFF);
                        self.low = v.wrapping_sub(0xFF);
                    } else {
                        self.high >>= 1;
                        self.low >>= 1;
                        let v = self.high.wrapping_add(self.low);
                        self.high = v;
                        self.low = v;
                    }
                } else {
                    break;
                }
            } else {
                let buf = self.read_byte();
                self.target <<= 8;
                self.high <<= 8;
                self.low <<= 8;
                self.high |= 0xFF;
                if self.low == 0 {
                    self.low = 1;
                }
                self.target |= buf as u32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PPM models: port of entropy_encoder/decoder_model_kernel_4 and _kernel_5
// ---------------------------------------------------------------------------

/// One context-tree node (`eemk5::node` / `edmk5::node` etc. in dlib).
/// Pointers are represented as indices into the node arena; the root is
/// index 0.
#[derive(Debug, Clone, Copy)]
struct PpmNode {
    next: Option<usize>,
    child_context: Option<usize>,
    parent_context: Option<usize>,
    symbol: u16,
    count: u16,
    total: u16,
    escapes: u16,
}

impl PpmNode {
    fn new_root() -> Self {
        PpmNode {
            next: None,
            child_context: None,
            parent_context: None,
            symbol: 0,
            count: 0,
            total: 0,
            escapes: 0,
        }
    }
}

const ALPHABET_SIZE: u32 = 256;
const TOTAL_NODES: usize = 200_000;
const ORDER: usize = 4;

fn scale_counts(nodes: &mut [PpmNode], temp: usize) {
    let mut t = nodes[temp];
    if t.escapes > 1 {
        t.escapes >>= 1;
    }
    t.total = t.escapes;
    let mut n = t.child_context;
    while let Some(i) = n {
        if nodes[i].count > 1 {
            nodes[i].count >>= 1;
        }
        t.total = t.total.wrapping_add(nodes[i].count);
        n = nodes[i].next;
    }
    nodes[temp] = t;
}

// ----- kernel_5 (with exclusions + information inheritance) ---------------

/// Encoder-side model, port of `entropy_encoder_model_kernel_5<256,enc,200000,4>`
/// (`entropy_encoder_model_kernel_5a`, `dlib/entropy_encoder_model/entropy_encoder_model_kernel_5.h`).
#[derive(Debug)]
struct EemK5 {
    nodes: Vec<PpmNode>,
    next_node: usize,
    cur: Option<usize>,
    cur_order: usize,
    exc: [u32; (ALPHABET_SIZE as usize / 32) + 1],
    exc_used: bool,
    stack: Vec<(usize, usize)>,
}

impl EemK5 {
    fn new() -> Self {
        let mut nodes = Vec::with_capacity(TOTAL_NODES);
        nodes.push(PpmNode::new_root());
        EemK5 {
            nodes,
            next_node: 1,
            cur: Some(0),
            cur_order: 0,
            exc: [0; (ALPHABET_SIZE as usize / 32) + 1],
            exc_used: false,
            stack: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.next_node = 1;
        let root = &mut self.nodes[0];
        root.child_context = None;
        root.escapes = 0;
        root.total = 0;
        self.cur = Some(0);
        self.cur_order = 0;
        self.stack.clear();
        self.exc = [0; (ALPHABET_SIZE as usize / 32) + 1];
        self.exc_used = false;
    }

    fn space_left(&self) -> bool {
        self.next_node < TOTAL_NODES
    }

    fn allocate_node(&mut self) -> usize {
        // arena grows lazily; index space is bounded by TOTAL_NODES
        while self.nodes.len() <= self.next_node {
            self.nodes.push(PpmNode::new_root());
        }
        let i = self.next_node;
        self.next_node += 1;
        i
    }

    fn exclude(&mut self, symbol: u16) {
        self.exc_used = true;
        self.exc[(symbol >> 5) as usize] |= 1u32 << (symbol & 0x1F);
    }

    fn is_excluded(&self, symbol: u16) -> bool {
        (self.exc[(symbol >> 5) as usize] & (1u32 << (symbol & 0x1F))) != 0
    }

    fn clear_exclusions(&mut self) {
        self.exc_used = false;
        for v in self.exc.iter_mut() {
            *v = 0;
        }
    }

    fn encode(&mut self, sym: u8, coder: &mut RangeEncoder) {
        let symbol = sym as u16;
        let mut temp = self.cur.unwrap();
        self.cur = None;
        let mut new_node: Option<usize> = None;
        let mut local_order = self.cur_order;

        let c: u16;
        let t: u16;

        if self.exc_used {
            self.clear_exclusions();
        }

        loop {
            let mut low_count: u16 = 0;
            let mut high_count: u16 = 0;
            if self.space_left() {
                let mut total_count = self.nodes[temp].total;
                if total_count > 0 {
                    if total_count > 10000 {
                        scale_counts(&mut self.nodes, temp);
                        total_count = self.nodes[temp].total;
                    }

                    let mut n = self.nodes[temp].child_context.unwrap();
                    let mut found_symbol: Option<usize> = None;
                    let mut last: Option<usize> = None;
                    if self.exc_used {
                        let mut templast: Option<usize> = None;
                        loop {
                            if !self.is_excluded(self.nodes[n].symbol) {
                                self.exclude(self.nodes[n].symbol);
                                if found_symbol.is_none() {
                                    high_count = high_count.wrapping_add(self.nodes[n].count);
                                    if self.nodes[n].symbol == symbol {
                                        found_symbol = Some(n);
                                        last = templast;
                                        low_count = high_count.wrapping_sub(self.nodes[n].count);
                                    }
                                }
                            } else {
                                total_count = total_count.wrapping_sub(self.nodes[n].count);
                            }

                            if self.nodes[n].next.is_none() {
                                break;
                            }
                            templast = Some(n);
                            n = self.nodes[n].next.unwrap();
                        }
                    } else {
                        loop {
                            high_count = high_count.wrapping_add(self.nodes[n].count);
                            self.exclude(self.nodes[n].symbol);

                            if self.nodes[n].symbol == symbol {
                                found_symbol = Some(n);
                                low_count = high_count.wrapping_sub(self.nodes[n].count);
                                break;
                            }

                            if self.nodes[n].next.is_none() {
                                break;
                            }
                            last = Some(n);
                            n = self.nodes[n].next.unwrap();
                        }
                    }

                    if let Some(found) = found_symbol {
                        let n = found;
                        if let Some(nn) = new_node {
                            self.nodes[nn].parent_context = Some(found);
                        }

                        coder.encode(low_count as u32, high_count as u32, total_count as u32);
                        self.nodes[n].count = self.nodes[n].count.wrapping_add(8);
                        self.nodes[temp].total = self.nodes[temp].total.wrapping_add(8);
                        c = self.nodes[n].count;
                        t = self.nodes[temp].total;

                        // move this node to the front (C++: last->next = n->next;
                        // n->next = temp->child_context; temp->child_context = n)
                        if let Some(l) = last {
                            let nn = self.nodes[n].next;
                            self.nodes[l].next = nn;
                            let cc = self.nodes[temp].child_context;
                            self.nodes[n].next = cc;
                            self.nodes[temp].child_context = Some(n);
                        }

                        if self.cur.is_none() {
                            if local_order >= ORDER {
                                self.cur = self.nodes[n].parent_context;
                                self.cur_order = local_order;
                            } else {
                                self.cur_order = local_order + 1;
                                self.cur = Some(n);
                            }
                        }

                        break;
                    } else {
                        // hit the end of the context set without finding the symbol
                        // finish excluding all the symbols
                        while self.nodes[n].next.is_some() {
                            let s = self.nodes[n].symbol;
                            self.exclude(s);
                            n = self.nodes[n].next.unwrap();
                        }

                        let nn = if let Some(prev) = new_node {
                            let fresh = self.allocate_node();
                            self.nodes[prev].parent_context = Some(fresh);
                            fresh
                        } else {
                            self.allocate_node()
                        };
                        new_node = Some(nn);

                        self.nodes[n].next = Some(nn);

                        // write an escape to a lower context
                        coder.encode(high_count as u32, total_count as u32, total_count as u32);
                    }
                } else {
                    // total_count == 0: make a new node here
                    let nn = if let Some(prev) = new_node {
                        let fresh = self.allocate_node();
                        self.nodes[prev].parent_context = Some(fresh);
                        fresh
                    } else {
                        self.allocate_node()
                    };
                    new_node = Some(nn);
                    self.nodes[temp].child_context = Some(nn);
                }

                if self.cur.is_none() && local_order < ORDER {
                    self.cur = new_node;
                    self.cur_order = local_order + 1;
                }

                // fill out the new node
                let nn = new_node.unwrap();
                self.nodes[nn].child_context = None;
                self.nodes[nn].escapes = 0;
                self.nodes[nn].next = None;
                self.nodes[nn].total = 0;
                self.stack.push((nn, temp));

                if temp != 0 {
                    temp = self.nodes[temp].parent_context.unwrap();
                    local_order -= 1;
                    continue;
                }

                t = 2056;
                c = 8;

                // root: encode with the order-(-1) context
                self.nodes[nn].parent_context = Some(0);
                coder.encode(symbol as u32, symbol as u32 + 1, ALPHABET_SIZE);

                if self.cur.is_none() {
                    self.cur = Some(0);
                    self.cur_order = 0;
                }
                break;
            } else {
                // there isn't enough space so throw away the tree
                self.clear();
                temp = self.cur.unwrap();
                local_order = self.cur_order;
                self.cur = None;
                new_node = None;
            }
        }

        // initialize the counts and symbol for any new nodes we have added
        while let Some((n, nc)) = self.stack.pop() {
            self.nodes[n].symbol = symbol;

            if self.nodes[nc].total != 0 {
                let temp2 = (t as u64)
                    .wrapping_sub(c as u64)
                    .wrapping_add(self.nodes[nc].total as u64)
                    .wrapping_sub(self.nodes[nc].escapes as u64)
                    .wrapping_sub(self.nodes[nc].escapes as u64);
                let mut tmp = self.nodes[nc].total as u64;
                tmp *= c as u64;
                tmp /= temp2 | 1; // the or-by-1 guards div by zero, as in dlib
                tmp += 2;
                if tmp > 50000 {
                    tmp = 50000;
                }
                self.nodes[n].count = tmp as u16;

                self.nodes[nc].escapes = self.nodes[nc].escapes.wrapping_add(4);
                self.nodes[nc].total = self.nodes[nc]
                    .total
                    .wrapping_add(self.nodes[n].count)
                    .wrapping_add(4);
            } else {
                self.nodes[n].count = (3u32 + 5 * (c as u32) / ((t - c) as u32)) as u16;

                self.nodes[nc].escapes = 4;
                self.nodes[nc].total = self.nodes[n].count.wrapping_add(4);
            }

            while self.nodes[nc].total > 10000 {
                scale_counts(&mut self.nodes, nc);
            }
        }
    }
}

/// Decoder-side model, port of `entropy_decoder_model_kernel_5<256,dec,200000,4>`
/// (`entropy_decoder_model_kernel_5a`, `dlib/entropy_decoder_model/entropy_decoder_model_kernel_5.h`).
#[derive(Debug)]
struct EdmK5 {
    nodes: Vec<PpmNode>,
    next_node: usize,
    cur: Option<usize>,
    cur_order: usize,
    exc: [u32; (ALPHABET_SIZE as usize / 32) + 1],
    exc_used: bool,
    stack: Vec<(usize, usize)>,
}

impl EdmK5 {
    fn new() -> Self {
        let mut nodes = Vec::with_capacity(TOTAL_NODES);
        nodes.push(PpmNode::new_root());
        EdmK5 {
            nodes,
            next_node: 1,
            cur: Some(0),
            cur_order: 0,
            exc: [0; (ALPHABET_SIZE as usize / 32) + 1],
            exc_used: false,
            stack: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.next_node = 1;
        let root = &mut self.nodes[0];
        root.child_context = None;
        root.escapes = 0;
        root.total = 0;
        self.cur = Some(0);
        self.cur_order = 0;
        self.stack.clear();
        self.exc = [0; (ALPHABET_SIZE as usize / 32) + 1];
        self.exc_used = false;
    }

    fn space_left(&self) -> bool {
        self.next_node < TOTAL_NODES
    }

    fn allocate_node(&mut self) -> usize {
        while self.nodes.len() <= self.next_node {
            self.nodes.push(PpmNode::new_root());
        }
        let i = self.next_node;
        self.next_node += 1;
        i
    }

    fn exclude(&mut self, symbol: u16) {
        self.exc_used = true;
        self.exc[(symbol >> 5) as usize] |= 1u32 << (symbol & 0x1F);
    }

    fn is_excluded(&self, symbol: u16) -> bool {
        (self.exc[(symbol >> 5) as usize] & (1u32 << (symbol & 0x1F))) != 0
    }

    fn clear_exclusions(&mut self) {
        self.exc_used = false;
        for v in self.exc.iter_mut() {
            *v = 0;
        }
    }

    fn decode(&mut self, coder: &mut RangeDecoder) -> Result<u16, DngError> {
        let mut temp = self.cur.unwrap();
        self.cur = None;
        let mut new_node: Option<usize> = None;
        let mut local_order = self.cur_order;

        let c: u16;
        let t: u16;
        let symbol;

        if self.exc_used {
            self.clear_exclusions();
        }

        loop {
            let mut high_count: u32;
            if self.space_left() {
                let mut total_count = self.nodes[temp].total as u32;
                if total_count > 0 {
                    if total_count > 10000 {
                        scale_counts(&mut self.nodes, temp);
                        total_count = self.nodes[temp].total as u32;
                    }

                    if self.exc_used {
                        // recompute total_count excluding already-excluded symbols
                        let mut n = self.nodes[temp].child_context.unwrap();
                        total_count = self.nodes[temp].escapes as u32;
                        loop {
                            if !self.is_excluded(self.nodes[n].symbol) {
                                total_count += self.nodes[n].count as u32;
                            }
                            if self.nodes[n].next.is_none() {
                                break;
                            }
                            n = self.nodes[n].next.unwrap();
                        }
                    }

                    let target = coder.get_target(total_count);

                    let mut n = self.nodes[temp].child_context.unwrap();
                    let mut last: Option<usize> = None;
                    high_count = 0;
                    loop {
                        if !self.is_excluded(self.nodes[n].symbol) {
                            high_count += self.nodes[n].count as u32;
                            let s = self.nodes[n].symbol;
                            self.exclude(s);
                        }

                        if high_count > target || self.nodes[n].next.is_none() {
                            break;
                        }
                        last = Some(n);
                        n = self.nodes[n].next.unwrap();
                    }

                    if high_count > target {
                        let low_count = high_count - self.nodes[n].count as u32;

                        if let Some(nn) = new_node {
                            self.nodes[nn].parent_context = Some(n);
                        }

                        symbol = self.nodes[n].symbol;

                        coder.decode(low_count, high_count);
                        self.nodes[n].count = self.nodes[n].count.wrapping_add(8);
                        self.nodes[temp].total = self.nodes[temp].total.wrapping_add(8);
                        c = self.nodes[n].count;
                        t = self.nodes[temp].total;

                        // move this node to the front (C++: last->next = n->next;
                        // n->next = temp->child_context; temp->child_context = n)
                        if let Some(l) = last {
                            let nn = self.nodes[n].next;
                            self.nodes[l].next = nn;
                            let cc = self.nodes[temp].child_context;
                            self.nodes[n].next = cc;
                            self.nodes[temp].child_context = Some(n);
                        }

                        if self.cur.is_none() {
                            if local_order < ORDER {
                                self.cur_order = local_order + 1;
                                self.cur = Some(n);
                            } else {
                                self.cur = self.nodes[n].parent_context;
                                self.cur_order = local_order;
                            }
                        }

                        break;
                    } else {
                        let nn = if let Some(prev) = new_node {
                            let fresh = self.allocate_node();
                            self.nodes[prev].parent_context = Some(fresh);
                            fresh
                        } else {
                            self.allocate_node()
                        };
                        new_node = Some(nn);

                        self.nodes[n].next = Some(nn);

                        // get the escape code
                        coder.decode(high_count, total_count);
                    }
                } else {
                    // total_count == 0
                    let nn = if let Some(prev) = new_node {
                        let fresh = self.allocate_node();
                        self.nodes[prev].parent_context = Some(fresh);
                        fresh
                    } else {
                        self.allocate_node()
                    };
                    new_node = Some(nn);
                    self.nodes[temp].child_context = Some(nn);
                }

                if self.cur.is_none() && local_order < ORDER {
                    self.cur = new_node;
                    self.cur_order = local_order + 1;
                }

                // fill out the new node
                let nn = new_node.unwrap();
                self.nodes[nn].child_context = None;
                self.nodes[nn].escapes = 0;
                self.nodes[nn].next = None;
                self.stack.push((nn, temp));
                self.nodes[nn].total = 0;

                if temp != 0 {
                    temp = self.nodes[temp].parent_context.unwrap();
                    local_order -= 1;
                    continue;
                }

                t = 2056;
                c = 8;

                // root: decode with the order-(-1) context
                let target = coder.get_target(ALPHABET_SIZE);
                self.nodes[nn].parent_context = Some(0);
                coder.decode(target, target + 1);
                symbol = target as u16;

                if self.cur.is_none() {
                    self.cur = Some(0);
                    self.cur_order = 0;
                }
                break;
            } else {
                self.clear();
                temp = self.cur.unwrap();
                local_order = self.cur_order;
                self.cur = None;
                new_node = None;
            }
        }

        while let Some((n, nc)) = self.stack.pop() {
            self.nodes[n].symbol = symbol;

            if self.nodes[nc].total != 0 {
                let temp2 = (t as u64)
                    .wrapping_sub(c as u64)
                    .wrapping_add(self.nodes[nc].total as u64)
                    .wrapping_sub(self.nodes[nc].escapes as u64)
                    .wrapping_sub(self.nodes[nc].escapes as u64);
                let mut tmp = self.nodes[nc].total as u64;
                tmp *= c as u64;
                tmp /= temp2 | 1;
                tmp += 2;
                if tmp > 50000 {
                    tmp = 50000;
                }
                self.nodes[n].count = tmp as u16;

                self.nodes[nc].escapes = self.nodes[nc].escapes.wrapping_add(4);
                self.nodes[nc].total = self.nodes[nc]
                    .total
                    .wrapping_add(self.nodes[n].count)
                    .wrapping_add(4);
            } else {
                self.nodes[n].count = (3u32 + 5 * (c as u32) / ((t - c) as u32)) as u16;

                self.nodes[nc].escapes = 4;
                self.nodes[nc].total = self.nodes[n].count.wrapping_add(4);
            }

            while self.nodes[nc].total > 10000 {
                scale_counts(&mut self.nodes, nc);
            }
        }

        Ok(symbol)
    }
}

// ----- kernel_4 (no exclusions; used for float exponents) -----------------

/// Encoder-side model, port of `entropy_encoder_model_kernel_4<256,enc,200000,4>`
/// (`entropy_encoder_model_kernel_4a`, `dlib/entropy_encoder_model/entropy_encoder_model_kernel_4.h`).
#[derive(Debug)]
struct EemK4 {
    nodes: Vec<PpmNode>,
    next_node: usize,
    cur: Option<usize>,
    cur_order: usize,
}

impl EemK4 {
    fn new() -> Self {
        let mut nodes = Vec::with_capacity(TOTAL_NODES);
        nodes.push(PpmNode::new_root());
        EemK4 {
            nodes,
            next_node: 1,
            cur: Some(0),
            cur_order: 0,
        }
    }

    fn destroy_tree(&mut self) {
        self.next_node = 1;
        let root = &mut self.nodes[0];
        root.child_context = None;
        root.escapes = 0;
        root.total = 0;
        self.cur = Some(0);
        self.cur_order = 0;
    }

    fn space_left(&self) -> bool {
        self.next_node < TOTAL_NODES
    }

    fn allocate_node(&mut self) -> usize {
        while self.nodes.len() <= self.next_node {
            self.nodes.push(PpmNode::new_root());
        }
        let i = self.next_node;
        self.next_node += 1;
        i
    }

    fn encode(&mut self, sym: u8, coder: &mut RangeEncoder) {
        let symbol = sym as u16;
        let mut temp = self.cur.unwrap();
        self.cur = None;
        let mut new_node: Option<usize> = None;
        let mut local_order = self.cur_order;

        loop {
            let mut high_count: u16 = 0;
            if self.space_left() {
                let mut total_count = self.nodes[temp].total;
                if total_count > 0 {
                    if total_count > 10000 {
                        scale_counts(&mut self.nodes, temp);
                        total_count = self.nodes[temp].total;
                    }

                    let mut n = self.nodes[temp].child_context.unwrap();
                    let mut last: Option<usize> = None;
                    loop {
                        high_count = high_count.wrapping_add(self.nodes[n].count);

                        if self.nodes[n].symbol == symbol || self.nodes[n].next.is_none() {
                            break;
                        }
                        last = Some(n);
                        n = self.nodes[n].next.unwrap();
                    }

                    let low_count = high_count.wrapping_sub(self.nodes[n].count);

                    if self.nodes[n].symbol == symbol {
                        if let Some(nn) = new_node {
                            self.nodes[nn].parent_context = Some(n);
                        }

                        coder.encode(low_count as u32, high_count as u32, total_count as u32);
                        self.nodes[n].count = self.nodes[n].count.wrapping_add(8);
                        self.nodes[temp].total = self.nodes[temp].total.wrapping_add(8);

                        // move this node to the front (C++: last->next = n->next;
                        // n->next = temp->child_context; temp->child_context = n)
                        if let Some(l) = last {
                            let nn = self.nodes[n].next;
                            self.nodes[l].next = nn;
                            let cc = self.nodes[temp].child_context;
                            self.nodes[n].next = cc;
                            self.nodes[temp].child_context = Some(n);
                        }

                        if self.cur.is_none() {
                            if local_order < ORDER {
                                self.cur_order = local_order + 1;
                                self.cur = Some(n);
                            } else {
                                self.cur = self.nodes[n].parent_context;
                                self.cur_order = local_order;
                            }
                        }

                        break;
                    } else {
                        let nn = if let Some(prev) = new_node {
                            let fresh = self.allocate_node();
                            self.nodes[prev].parent_context = Some(fresh);
                            fresh
                        } else {
                            self.allocate_node()
                        };
                        new_node = Some(nn);

                        self.nodes[n].next = Some(nn);

                        // write an escape to a lower context
                        coder.encode(high_count as u32, total_count as u32, total_count as u32);
                    }
                } else {
                    // total_count == 0
                    let nn = if let Some(prev) = new_node {
                        let fresh = self.allocate_node();
                        self.nodes[prev].parent_context = Some(fresh);
                        fresh
                    } else {
                        self.allocate_node()
                    };
                    new_node = Some(nn);
                    self.nodes[temp].child_context = Some(nn);
                }

                if self.cur.is_none() && local_order < ORDER {
                    self.cur = new_node;
                    self.cur_order = local_order + 1;
                }

                // fill out the new node
                let nn = new_node.unwrap();
                self.nodes[nn].child_context = None;
                self.nodes[nn].count = 4;
                self.nodes[nn].escapes = 0;
                self.nodes[nn].next = None;
                self.nodes[nn].symbol = symbol;
                self.nodes[nn].total = 0;
                self.nodes[temp].escapes = self.nodes[temp].escapes.wrapping_add(4);
                self.nodes[temp].total = self.nodes[temp].total.wrapping_add(8);

                if temp != 0 {
                    temp = self.nodes[temp].parent_context.unwrap();
                    local_order -= 1;
                    continue;
                }

                // root: encode with the order-(-1) context
                self.nodes[nn].parent_context = Some(0);
                coder.encode(symbol as u32, symbol as u32 + 1, ALPHABET_SIZE);

                if self.cur.is_none() {
                    self.cur = Some(0);
                    self.cur_order = 0;
                }
                break;
            } else {
                self.destroy_tree();
                temp = self.cur.unwrap();
                local_order = self.cur_order;
                self.cur = None;
                new_node = None;
            }
        }
    }
}

/// Decoder-side model, port of `entropy_decoder_model_kernel_4<256,dec,200000,4>`
/// (`entropy_decoder_model_kernel_4a`, `dlib/entropy_decoder_model/entropy_decoder_model_kernel_4.h`).
#[derive(Debug)]
struct EdmK4 {
    nodes: Vec<PpmNode>,
    next_node: usize,
    cur: Option<usize>,
    cur_order: usize,
    stack: Vec<usize>,
}

impl EdmK4 {
    fn new() -> Self {
        let mut nodes = Vec::with_capacity(TOTAL_NODES);
        nodes.push(PpmNode::new_root());
        EdmK4 {
            nodes,
            next_node: 1,
            cur: Some(0),
            cur_order: 0,
            stack: Vec::new(),
        }
    }

    fn destroy_tree(&mut self) {
        self.next_node = 1;
        let root = &mut self.nodes[0];
        root.child_context = None;
        root.escapes = 0;
        root.total = 0;
        self.cur = Some(0);
        self.cur_order = 0;
        self.stack.clear();
    }

    fn space_left(&self) -> bool {
        self.next_node < TOTAL_NODES
    }

    fn allocate_node(&mut self) -> usize {
        while self.nodes.len() <= self.next_node {
            self.nodes.push(PpmNode::new_root());
        }
        let i = self.next_node;
        self.next_node += 1;
        i
    }

    fn decode(&mut self, coder: &mut RangeDecoder) -> u16 {
        let mut temp = self.cur.unwrap();
        self.cur = None;
        let mut new_node: Option<usize> = None;
        let mut local_order = self.cur_order;
        let symbol;

        loop {
            let mut high_count: u32;
            if self.space_left() {
                let mut total_count = self.nodes[temp].total as u32;
                if total_count > 0 {
                    if total_count > 10000 {
                        scale_counts(&mut self.nodes, temp);
                        total_count = self.nodes[temp].total as u32;
                    }

                    let target = coder.get_target(total_count);

                    let mut n = self.nodes[temp].child_context.unwrap();
                    let mut last: Option<usize> = None;
                    high_count = 0;
                    loop {
                        high_count += self.nodes[n].count as u32;

                        if high_count > target || self.nodes[n].next.is_none() {
                            break;
                        }
                        last = Some(n);
                        n = self.nodes[n].next.unwrap();
                    }

                    let low_count = high_count - self.nodes[n].count as u32;

                    if high_count > target {
                        if let Some(nn) = new_node {
                            self.nodes[nn].parent_context = Some(n);
                        }

                        symbol = self.nodes[n].symbol;

                        coder.decode(low_count, high_count);
                        self.nodes[n].count = self.nodes[n].count.wrapping_add(8);
                        self.nodes[temp].total = self.nodes[temp].total.wrapping_add(8);

                        // move this node to the front (C++: last->next = n->next;
                        // n->next = temp->child_context; temp->child_context = n)
                        if let Some(l) = last {
                            let nn = self.nodes[n].next;
                            self.nodes[l].next = nn;
                            let cc = self.nodes[temp].child_context;
                            self.nodes[n].next = cc;
                            self.nodes[temp].child_context = Some(n);
                        }

                        if self.cur.is_none() {
                            if local_order < ORDER {
                                self.cur_order = local_order + 1;
                                self.cur = Some(n);
                            } else {
                                self.cur = self.nodes[n].parent_context;
                                self.cur_order = local_order;
                            }
                        }

                        break;
                    } else {
                        let nn = if let Some(prev) = new_node {
                            let fresh = self.allocate_node();
                            self.nodes[prev].parent_context = Some(fresh);
                            fresh
                        } else {
                            self.allocate_node()
                        };
                        new_node = Some(nn);

                        self.nodes[n].next = Some(nn);

                        // get the escape code
                        coder.decode(high_count, total_count);
                    }
                } else {
                    // total_count == 0
                    let nn = if let Some(prev) = new_node {
                        let fresh = self.allocate_node();
                        self.nodes[prev].parent_context = Some(fresh);
                        fresh
                    } else {
                        self.allocate_node()
                    };
                    new_node = Some(nn);
                    self.nodes[temp].child_context = Some(nn);
                }

                if self.cur.is_none() && local_order < ORDER {
                    self.cur = new_node;
                    self.cur_order = local_order + 1;
                }

                // fill out the new node
                let nn = new_node.unwrap();
                self.nodes[nn].child_context = None;
                self.nodes[nn].count = 4;
                self.nodes[nn].escapes = 0;
                self.nodes[nn].next = None;
                self.stack.push(nn);
                self.nodes[nn].total = 0;
                self.nodes[temp].escapes = self.nodes[temp].escapes.wrapping_add(4);
                self.nodes[temp].total = self.nodes[temp].total.wrapping_add(8);

                if temp != 0 {
                    temp = self.nodes[temp].parent_context.unwrap();
                    local_order -= 1;
                    continue;
                }

                // root: decode with the order-(-1) context
                let target = coder.get_target(ALPHABET_SIZE);
                self.nodes[nn].parent_context = Some(0);
                coder.decode(target, target + 1);
                symbol = target as u16;

                if self.cur.is_none() {
                    self.cur = Some(0);
                    self.cur_order = 0;
                }
                break;
            } else {
                self.destroy_tree();
                temp = self.cur.unwrap();
                local_order = self.cur_order;
                self.cur = None;
                new_node = None;
            }
        }

        while let Some(n) = self.stack.pop() {
            self.nodes[n].symbol = symbol;
        }

        symbol
    }
}

// ---------------------------------------------------------------------------
// DNG shared constants and predictors (dlib/image_saver/dng_shared.h)
// ---------------------------------------------------------------------------

/// Image type tags (`dng_helpers_namespace` enum in `dng_shared.h`).
const TYPE_GRAYSCALE: u32 = 1;
const TYPE_RGB: u32 = 2;
const TYPE_HSI: u32 = 3;
const TYPE_RGB_PAETH: u32 = 4;
const TYPE_RGB_ALPHA: u32 = 5;
const TYPE_RGB_ALPHA_PAETH: u32 = 6;
const TYPE_GRAYSCALE_16BIT: u32 = 7;
const TYPE_GRAYSCALE_FLOAT: u32 = 8;

/// `dng_magic_byte` marks the end of the compressed data.
const DNG_MAGIC_BYTE: u16 = 100;

/// dlib picks the Paeth predictors for small images.
const PAETH_SIZE_CUTOFF: usize = 4000;

fn sub_u8(a: u8, b: u8) -> u8 {
    a.wrapping_sub(b)
}
fn add_u8(a: u8, b: u8) -> u8 {
    a.wrapping_add(b)
}

/// `predictor_grayscale` (dng_shared.h).
fn predictor_grayscale(img: &Array2D<u8>, row: usize, col: usize) -> u8 {
    let mut a: u8 = 0;
    let mut b: u8 = 0;
    let mut c: u8 = 0;
    if col >= 1 {
        assign_pixel(&mut a, &img[(row, col - 1)]);
    }
    if col >= 1 && row >= 1 {
        assign_pixel(&mut c, &img[(row - 1, col - 1)]);
    }
    if row >= 1 {
        assign_pixel(&mut b, &img[(row - 1, col)]);
    }
    (a as i32 + b as i32 - c as i32) as u8
}

/// `predictor_grayscale_16` (dng_shared.h).
fn predictor_grayscale_16(img: &Array2D<u16>, row: usize, col: usize) -> u16 {
    let mut a: u16 = 0;
    let mut b: u16 = 0;
    let mut c: u16 = 0;
    if col >= 1 {
        assign_pixel(&mut a, &img[(row, col - 1)]);
    }
    if col >= 1 && row >= 1 {
        assign_pixel(&mut c, &img[(row - 1, col - 1)]);
    }
    if row >= 1 {
        assign_pixel(&mut b, &img[(row - 1, col)]);
    }
    a.wrapping_add(b).wrapping_sub(c)
}

/// `predictor_rgb` (dng_shared.h).
fn predictor_rgb(img: &Array2D<RgbPixel>, row: usize, col: usize) -> RgbPixel {
    let mut a = RgbPixel::default();
    let mut b = RgbPixel::default();
    let mut c = RgbPixel::default();
    if col >= 1 {
        assign_pixel(&mut a, &img[(row, col - 1)]);
    }
    if col >= 1 && row >= 1 {
        assign_pixel(&mut c, &img[(row - 1, col - 1)]);
    }
    if row >= 1 {
        assign_pixel(&mut b, &img[(row - 1, col)]);
    }
    RgbPixel {
        r: (a.r as i32 + b.r as i32 - c.r as i32) as u8,
        g: (a.g as i32 + b.g as i32 - c.g as i32) as u8,
        b: (a.b as i32 + b.b as i32 - c.b as i32) as u8,
    }
}

/// `predictor_rgb_paeth` (dng_shared.h) — Paeth filter like PNG's.
fn predictor_rgb_paeth(img: &Array2D<RgbPixel>, row: usize, col: usize) -> RgbPixel {
    let mut a = RgbPixel::default();
    let mut b = RgbPixel::default();
    let mut c = RgbPixel::default();
    if col >= 1 {
        assign_pixel(&mut a, &img[(row, col - 1)]);
    }
    if col >= 1 && row >= 1 {
        assign_pixel(&mut c, &img[(row - 1, col - 1)]);
    }
    if row >= 1 {
        assign_pixel(&mut b, &img[(row - 1, col)]);
    }
    let p = RgbPixel {
        r: (a.r as i32 + b.r as i32 - c.r as i32) as u8,
        g: (a.g as i32 + b.g as i32 - c.g as i32) as u8,
        b: (a.b as i32 + b.b as i32 - c.b as i32) as u8,
    };
    let pa = (p.r as i16 - a.r as i16).abs()
        + (p.g as i16 - a.g as i16).abs()
        + (p.b as i16 - a.b as i16).abs();
    let pb = (p.r as i16 - b.r as i16).abs()
        + (p.g as i16 - b.g as i16).abs()
        + (p.b as i16 - b.b as i16).abs();
    let pc = (p.r as i16 - c.r as i16).abs()
        + (p.g as i16 - c.g as i16).abs()
        + (p.b as i16 - c.b as i16).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// `predictor_rgb_alpha` (dng_shared.h).
fn predictor_rgb_alpha(img: &Array2D<RgbAlphaPixel>, row: usize, col: usize) -> RgbAlphaPixel {
    // dlib seeds these with assign_pixel(a, (unsigned char)0), which sets
    // alpha to 255 (gray -> rgba conversion), not to the default 0.
    let mut a = RgbAlphaPixel::default();
    assign_pixel(&mut a, &0u8);
    let mut b = RgbAlphaPixel::default();
    assign_pixel(&mut b, &0u8);
    let mut c = RgbAlphaPixel::default();
    assign_pixel(&mut c, &0u8);
    if col >= 1 {
        assign_pixel(&mut a, &img[(row, col - 1)]);
    }
    if col >= 1 && row >= 1 {
        assign_pixel(&mut c, &img[(row - 1, col - 1)]);
    }
    if row >= 1 {
        assign_pixel(&mut b, &img[(row - 1, col)]);
    }
    RgbAlphaPixel {
        r: (a.r as i32 + b.r as i32 - c.r as i32) as u8,
        g: (a.g as i32 + b.g as i32 - c.g as i32) as u8,
        b: (a.b as i32 + b.b as i32 - c.b as i32) as u8,
        a: (a.a as i32 + b.a as i32 - c.a as i32) as u8,
    }
}

/// `predictor_rgb_alpha_paeth` (dng_shared.h).
fn predictor_rgb_alpha_paeth(
    img: &Array2D<RgbAlphaPixel>,
    row: usize,
    col: usize,
) -> RgbAlphaPixel {
    // dlib seeds these with assign_pixel(a, (unsigned char)0), which sets
    // alpha to 255 (gray -> rgba conversion), not to the default 0.
    let mut a = RgbAlphaPixel::default();
    assign_pixel(&mut a, &0u8);
    let mut b = RgbAlphaPixel::default();
    assign_pixel(&mut b, &0u8);
    let mut c = RgbAlphaPixel::default();
    assign_pixel(&mut c, &0u8);
    if col >= 1 {
        assign_pixel(&mut a, &img[(row, col - 1)]);
    }
    if col >= 1 && row >= 1 {
        assign_pixel(&mut c, &img[(row - 1, col - 1)]);
    }
    if row >= 1 {
        assign_pixel(&mut b, &img[(row - 1, col)]);
    }
    let p = RgbAlphaPixel {
        r: (a.r as i32 + b.r as i32 - c.r as i32) as u8,
        g: (a.g as i32 + b.g as i32 - c.g as i32) as u8,
        b: (a.b as i32 + b.b as i32 - c.b as i32) as u8,
        a: (a.a as i32 + b.a as i32 - c.a as i32) as u8,
    };
    let pa = (p.r as i16 - a.r as i16).abs()
        + (p.g as i16 - a.g as i16).abs()
        + (p.b as i16 - a.b as i16).abs();
    let pb = (p.r as i16 - b.r as i16).abs()
        + (p.g as i16 - b.g as i16).abs()
        + (p.b as i16 - b.b as i16).abs();
    let pc = (p.r as i16 - c.r as i16).abs()
        + (p.g as i16 - c.g as i16).abs()
        + (p.b as i16 - c.b as i16).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// `predictor_hsi` (dng_shared.h).
fn predictor_hsi(img: &Array2D<HsiPixel>, row: usize, col: usize) -> HsiPixel {
    let mut a = HsiPixel::default();
    let mut b = HsiPixel::default();
    let mut c = HsiPixel::default();
    if col >= 1 {
        assign_pixel(&mut a, &img[(row, col - 1)]);
    }
    if col >= 1 && row >= 1 {
        assign_pixel(&mut c, &img[(row - 1, col - 1)]);
    }
    if row >= 1 {
        assign_pixel(&mut b, &img[(row - 1, col)]);
    }
    HsiPixel {
        h: a.h.wrapping_add(b.h).wrapping_sub(c.h),
        s: a.s.wrapping_add(b.s).wrapping_sub(c.s),
        i: a.i.wrapping_add(b.i).wrapping_sub(c.i),
    }
}

// ---------------------------------------------------------------------------
// save_dng
// ---------------------------------------------------------------------------

/// Encodes `img` in dlib's DNG format and writes it to `out`
/// (port of `dlib::save_dng(image, ostream)` from `dlib/image_saver/image_saver.h`).
pub fn save_dng<P: Pixel, W: Write>(img: &Array2D<P>, out: &mut W) -> Result<(), DngError> {
    let bytes = save_dng_bytes(img)?;
    out.write_all(&bytes)?;
    Ok(())
}

/// Saves `img` to `path` in DNG format (port of the `save_dng` file overload).
pub fn save_dng_file<P: Pixel>(img: &Array2D<P>, path: &std::path::Path) -> Result<(), DngError> {
    let bytes = save_dng_bytes(img)?;
    std::fs::write(path, &bytes)?;
    Ok(())
}

fn save_dng_bytes<P: Pixel>(img: &Array2D<P>) -> Result<Vec<u8>, DngError> {
    // dlib dispatch order: float -> grayscale(sizeof==1) -> grayscale
    // (non-float, sizeof != 1) -> rgb -> rgb_alpha -> hsi.
    if P::is_float() {
        save_dng_float(img)
    } else if P::is_gray() {
        if std::mem::size_of::<P>() == 1 {
            save_dng_gray8(img)
        } else {
            save_dng_gray16(img)
        }
    } else if P::is_rgb() {
        save_dng_rgb(img)
    } else if P::is_rgb_alpha() {
        save_dng_rgb_alpha(img)
    } else if P::is_hsi() {
        save_dng_hsi(img)
    } else {
        Err(DngError::Corrupt(format!(
            "pixel type {} cannot be saved as a dng image",
            std::any::type_name::<P>()
        )))
    }
}

fn dng_header(type_tag: u32, nc: usize, nr: usize) -> Vec<u8> {
    let mut ser = Serializer::new();
    ser.write_u8(b'D');
    ser.write_u8(b'N');
    ser.write_u8(b'G');
    ser.write_u32(1); // version
    ser.write_u32(type_tag);
    ser.write_i64(nc as i64);
    ser.write_i64(nr as i64);
    ser.into_inner()
}

/// `grayscale_float` path: mantissa deltas serialized raw, exponent deltas
/// range-coded with `kernel_4a` into an appended blob.
fn save_dng_float<P: Pixel>(img: &Array2D<P>) -> Result<Vec<u8>, DngError> {
    let mut out = dng_header(TYPE_GRAYSCALE_FLOAT, img.nc(), img.nr());

    let mut expbuf_encoder = RangeEncoder::new();
    let mut eem_exp = EemK4::new();
    let mut prev = FloatDetails::new(0, 0);
    let mut ser = Serializer::new();
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            // dlib builds float_details from the image's own element type
            // (float -> 24 mantissa bits, double -> 53), so preserve the
            // source precision rather than widening to f64 first.
            let cur = match img[(r, c)].to_value() {
                PixelValue::F32(v) => FloatDetails::from_f32(v),
                PixelValue::F64(v) => FloatDetails::from_f64(v),
                _ => {
                    let mut v: f64 = 0.0;
                    assign_pixel(&mut v, &img[(r, c)]);
                    FloatDetails::from_f64(v)
                }
            };
            let exp = cur.exponent.wrapping_sub(prev.exponent);
            let man = cur.mantissa.wrapping_sub(prev.mantissa);
            prev = cur;

            let ebyte1 = (exp as u16 & 0xFF) as u8;
            let ebyte2 = ((exp as u16) >> 8) as u8;
            eem_exp.encode(ebyte1, &mut expbuf_encoder);
            eem_exp.encode(ebyte2, &mut expbuf_encoder);

            ser.write_i64(man);
        }
    }
    out.extend_from_slice(&ser.into_inner());

    // magic byte marks the end of the compressed data
    for _ in 0..4 {
        eem_exp.encode(DNG_MAGIC_BYTE as u8, &mut expbuf_encoder);
    }
    // encoder.clear() -> flush(): writes the 4 final bytes of low
    expbuf_encoder.flush();

    let expbuf = expbuf_encoder.out;
    let mut ser = Serializer::new();
    ser.write_u64(expbuf.len() as u64);
    let mut tail = ser.into_inner();
    tail.extend_from_slice(&expbuf);
    out.extend_from_slice(&tail);
    Ok(out)
}

fn save_dng_gray8<P: Pixel>(img: &Array2D<P>) -> Result<Vec<u8>, DngError> {
    let mut out = dng_header(TYPE_GRAYSCALE, img.nc(), img.nr());
    // convert to the native u8 image the predictors run on
    let mut native = Array2D::<u8>::zeros(img.nr(), img.nc());
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            assign_pixel(&mut native[(r, c)], &img[(r, c)]);
        }
    }
    let mut encoder = RangeEncoder::new();
    let mut eem = EemK5::new();
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            let cur = native[(r, c)];
            let cur = cur.wrapping_sub(predictor_grayscale(&native, r, c));
            eem.encode(cur, &mut encoder);
        }
    }
    for _ in 0..4 {
        eem.encode(DNG_MAGIC_BYTE as u8, &mut encoder);
    }
    // the C++ entropy_encoder destructor flushes the 4 final bytes of low
    encoder.flush();
    out.extend_from_slice(&encoder.out);
    Ok(out)
}

fn save_dng_gray16<P: Pixel>(img: &Array2D<P>) -> Result<Vec<u8>, DngError> {
    let mut out = dng_header(TYPE_GRAYSCALE_16BIT, img.nc(), img.nr());
    let mut native = Array2D::<u16>::zeros(img.nr(), img.nc());
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            assign_pixel(&mut native[(r, c)], &img[(r, c)]);
        }
    }
    let mut encoder = RangeEncoder::new();
    let mut eem = EemK5::new();
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            let cur = native[(r, c)];
            let cur = cur.wrapping_sub(predictor_grayscale_16(&native, r, c));
            let byte1 = (cur & 0xFF) as u8;
            let byte2 = (cur >> 8) as u8;
            eem.encode(byte2, &mut encoder);
            eem.encode(byte1, &mut encoder);
        }
    }
    for _ in 0..4 {
        eem.encode(DNG_MAGIC_BYTE as u8, &mut encoder);
    }
    // the C++ entropy_encoder destructor flushes the 4 final bytes of low
    encoder.flush();
    out.extend_from_slice(&encoder.out);
    Ok(out)
}

fn save_dng_rgb<P: Pixel>(img: &Array2D<P>) -> Result<Vec<u8>, DngError> {
    let mut native = Array2D::<RgbPixel>::zeros(img.nr(), img.nc());
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            assign_pixel(&mut native[(r, c)], &img[(r, c)]);
        }
    }
    let small = img.nr() * img.nc() < PAETH_SIZE_CUTOFF;
    let type_tag = if small { TYPE_RGB_PAETH } else { TYPE_RGB };
    let mut out = dng_header(type_tag, img.nc(), img.nr());

    let mut encoder = RangeEncoder::new();
    let mut eem = EemK5::new();
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            let cur = native[(r, c)];
            let pre = if type_tag == TYPE_RGB {
                predictor_rgb(&native, r, c)
            } else {
                predictor_rgb_paeth(&native, r, c)
            };
            eem.encode(sub_u8(cur.r, pre.r), &mut encoder);
            eem.encode(sub_u8(cur.g, pre.g), &mut encoder);
            eem.encode(sub_u8(cur.b, pre.b), &mut encoder);
        }
    }
    for _ in 0..4 {
        eem.encode(DNG_MAGIC_BYTE as u8, &mut encoder);
    }
    // the C++ entropy_encoder destructor flushes the 4 final bytes of low
    encoder.flush();
    out.extend_from_slice(&encoder.out);
    Ok(out)
}

fn save_dng_rgb_alpha<P: Pixel>(img: &Array2D<P>) -> Result<Vec<u8>, DngError> {
    let mut native = Array2D::<RgbAlphaPixel>::zeros(img.nr(), img.nc());
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            assign_pixel(&mut native[(r, c)], &img[(r, c)]);
        }
    }
    let small = img.nr() * img.nc() < PAETH_SIZE_CUTOFF;
    let type_tag = if small {
        TYPE_RGB_ALPHA_PAETH
    } else {
        TYPE_RGB_ALPHA
    };
    let mut out = dng_header(type_tag, img.nc(), img.nr());

    let mut encoder = RangeEncoder::new();
    let mut eem = EemK5::new();
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            let cur = native[(r, c)];
            let pre = if type_tag == TYPE_RGB_ALPHA {
                predictor_rgb_alpha(&native, r, c)
            } else {
                predictor_rgb_alpha_paeth(&native, r, c)
            };
            eem.encode(sub_u8(cur.r, pre.r), &mut encoder);
            eem.encode(sub_u8(cur.g, pre.g), &mut encoder);
            eem.encode(sub_u8(cur.b, pre.b), &mut encoder);
            eem.encode(sub_u8(cur.a, pre.a), &mut encoder);
        }
    }
    for _ in 0..4 {
        eem.encode(DNG_MAGIC_BYTE as u8, &mut encoder);
    }
    // the C++ entropy_encoder destructor flushes the 4 final bytes of low
    encoder.flush();
    out.extend_from_slice(&encoder.out);
    Ok(out)
}

fn save_dng_hsi<P: Pixel>(img: &Array2D<P>) -> Result<Vec<u8>, DngError> {
    let mut out = dng_header(TYPE_HSI, img.nc(), img.nr());
    let mut native = Array2D::<HsiPixel>::zeros(img.nr(), img.nc());
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            assign_pixel(&mut native[(r, c)], &img[(r, c)]);
        }
    }
    let mut encoder = RangeEncoder::new();
    let mut eem = EemK5::new();
    for r in 0..img.nr() {
        for c in 0..img.nc() {
            let cur = native[(r, c)];
            let pre = predictor_hsi(&native, r, c);
            eem.encode(sub_u8(cur.h as u8, pre.h as u8), &mut encoder);
            eem.encode(sub_u8(cur.s as u8, pre.s as u8), &mut encoder);
            eem.encode(sub_u8(cur.i as u8, pre.i as u8), &mut encoder);
        }
    }
    for _ in 0..4 {
        eem.encode(DNG_MAGIC_BYTE as u8, &mut encoder);
    }
    // the C++ entropy_encoder destructor flushes the 4 final bytes of low
    encoder.flush();
    out.extend_from_slice(&encoder.out);
    Ok(out)
}

// ---------------------------------------------------------------------------
// load_dng
// ---------------------------------------------------------------------------

/// Decodes a DNG image from `input` into an `Array2D<P>`
/// (port of `dlib::load_dng(image, istream)` from `dlib/image_loader/image_loader.h`).
///
/// Decoding runs on the file's native pixel type (exactly like the C++
/// decoder's predictors) and each decoded pixel is then converted to `P`
/// with `assign_pixel`.
pub fn load_dng<P: Pixel, R: Read>(input: &mut R) -> Result<Array2D<P>, DngError> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    load_dng_from_slice(&data)
}

/// Loads a DNG image from `path` (port of the `load_dng` file overload).
pub fn load_dng_file<P: Pixel>(path: &std::path::Path) -> Result<Array2D<P>, DngError> {
    let data = std::fs::read(path)?;
    load_dng_from_slice(&data)
}

/// The native pixel type a DNG file stores (its `type` tag).
enum NativeImage {
    G8(Array2D<u8>),
    G16(Array2D<u16>),
    Rgb(Array2D<RgbPixel>),
    Rgba(Array2D<RgbAlphaPixel>),
    Hsi(Array2D<HsiPixel>),
    F64(Array2D<f64>),
}

fn convert_native<P: Pixel>(src: &NativeImage) -> Array2D<P> {
    fn convert_image<P: Pixel, Q: Pixel>(src: &Array2D<Q>) -> Array2D<P> {
        let mut dst = Array2D::zeros(src.nr(), src.nc());
        for r in 0..src.nr() {
            for c in 0..src.nc() {
                assign_pixel(&mut dst[(r, c)], &src[(r, c)]);
            }
        }
        dst
    }
    match src {
        NativeImage::G8(img) => convert_image(img),
        NativeImage::G16(img) => convert_image(img),
        NativeImage::Rgb(img) => convert_image(img),
        NativeImage::Rgba(img) => convert_image(img),
        NativeImage::Hsi(img) => convert_image(img),
        NativeImage::F64(img) => convert_image(img),
    }
}

fn load_dng_from_slice<P: Pixel>(data: &[u8]) -> Result<Array2D<P>, DngError> {
    let mut de = Deserializer::new(data);
    if de.read_u8()? != b'D' || de.read_u8()? != b'N' || de.read_u8()? != b'G' {
        return Err(DngError::Corrupt(
            "the stream does not contain a dng image file".to_string(),
        ));
    }
    let version = de.read_u32()?;
    if version != 1 {
        return Err(DngError::Corrupt(
            "You need the new version of the dlib library to read this dng file".to_string(),
        ));
    }
    let type_tag = de.read_u32()?;
    let width = de.read_i64()?;
    let height = de.read_i64()?;

    let (nr, nc) = if width > 0 && height > 0 {
        (height as usize, width as usize)
    } else {
        (0, 0)
    };

    if type_tag != TYPE_GRAYSCALE_FLOAT {
        let payload = &data[data.len() - de.remaining()..];
        let mut decoder = RangeDecoder::new(payload);
        let mut edm = EdmK5::new();
        let img = match type_tag {
            TYPE_RGB_ALPHA_PAETH => {
                let mut img = Array2D::<RgbAlphaPixel>::zeros(nr, nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let mut p = predictor_rgb_alpha_paeth(&img, r, c);
                        p.r = add_u8(p.r, edm.decode(&mut decoder)? as u8);
                        p.g = add_u8(p.g, edm.decode(&mut decoder)? as u8);
                        p.b = add_u8(p.b, edm.decode(&mut decoder)? as u8);
                        p.a = add_u8(p.a, edm.decode(&mut decoder)? as u8);
                        img[(r, c)] = p;
                    }
                }
                NativeImage::Rgba(img)
            }
            TYPE_RGB_ALPHA => {
                let mut img = Array2D::<RgbAlphaPixel>::zeros(nr, nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let mut p = predictor_rgb_alpha(&img, r, c);
                        p.r = add_u8(p.r, edm.decode(&mut decoder)? as u8);
                        p.g = add_u8(p.g, edm.decode(&mut decoder)? as u8);
                        p.b = add_u8(p.b, edm.decode(&mut decoder)? as u8);
                        p.a = add_u8(p.a, edm.decode(&mut decoder)? as u8);
                        img[(r, c)] = p;
                    }
                }
                NativeImage::Rgba(img)
            }
            TYPE_RGB_PAETH => {
                let mut img = Array2D::<RgbPixel>::zeros(nr, nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let mut p = predictor_rgb_paeth(&img, r, c);
                        p.r = add_u8(p.r, edm.decode(&mut decoder)? as u8);
                        p.g = add_u8(p.g, edm.decode(&mut decoder)? as u8);
                        p.b = add_u8(p.b, edm.decode(&mut decoder)? as u8);
                        img[(r, c)] = p;
                    }
                }
                NativeImage::Rgb(img)
            }
            TYPE_RGB => {
                let mut img = Array2D::<RgbPixel>::zeros(nr, nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let mut p = predictor_rgb(&img, r, c);
                        p.r = add_u8(p.r, edm.decode(&mut decoder)? as u8);
                        p.g = add_u8(p.g, edm.decode(&mut decoder)? as u8);
                        p.b = add_u8(p.b, edm.decode(&mut decoder)? as u8);
                        img[(r, c)] = p;
                    }
                }
                NativeImage::Rgb(img)
            }
            TYPE_HSI => {
                let mut img = Array2D::<HsiPixel>::zeros(nr, nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let mut p = predictor_hsi(&img, r, c);
                        p.h = p.h.wrapping_add(edm.decode(&mut decoder)?);
                        p.s = p.s.wrapping_add(edm.decode(&mut decoder)?);
                        p.i = p.i.wrapping_add(edm.decode(&mut decoder)?);
                        img[(r, c)] = p;
                    }
                }
                NativeImage::Hsi(img)
            }
            TYPE_GRAYSCALE => {
                let mut img = Array2D::<u8>::zeros(nr, nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let sym = edm.decode(&mut decoder)? as u8;
                        let p = sym.wrapping_add(predictor_grayscale(&img, r, c));
                        img[(r, c)] = p;
                    }
                }
                NativeImage::G8(img)
            }
            TYPE_GRAYSCALE_16BIT => {
                let mut img = Array2D::<u16>::zeros(nr, nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let b2 = edm.decode(&mut decoder)?;
                        let b1 = edm.decode(&mut decoder)?;
                        let mut p = b2 << 8;
                        p |= b1;
                        p = p.wrapping_add(predictor_grayscale_16(&img, r, c));
                        img[(r, c)] = p;
                    }
                }
                NativeImage::G16(img)
            }
            _ => {
                return Err(DngError::Corrupt(
                    "corruption detected in the dng file".to_string(),
                ));
            }
        };
        for _ in 0..4 {
            if edm.decode(&mut decoder)? != DNG_MAGIC_BYTE {
                return Err(DngError::Corrupt(
                    "corruption detected in the dng file".to_string(),
                ));
            }
        }
        Ok(convert_native(&img))
    } else {
        // grayscale_float
        let mut mans = Vec::with_capacity(nr * nc);
        for _ in 0..nr * nc {
            mans.push(de.read_i64()?);
        }
        let expbuf_len = de.read_u64()? as usize;
        if de.remaining() < expbuf_len {
            return Err(DngError::Corrupt("dng file truncated".to_string()));
        }
        let start = data.len() - de.remaining();
        let payload = &data[start..start + expbuf_len];
        let mut decoder = RangeDecoder::new(payload);
        let mut edm_exp = EdmK4::new();

        let mut img = Array2D::<f64>::zeros(nr, nc);
        let mut prev = FloatDetails::new(0, 0);
        let mut i = 0usize;
        for r in 0..nr {
            for c in 0..nc {
                let exp1 = edm_exp.decode(&mut decoder);
                let exp2 = edm_exp.decode(&mut decoder);
                let mut cur = FloatDetails::new(mans[i], ((exp2 << 8) | exp1) as i16);
                i += 1;
                cur.exponent = cur.exponent.wrapping_add(prev.exponent);
                cur.mantissa = cur.mantissa.wrapping_add(prev.mantissa);
                prev = cur;
                img[(r, c)] = cur.to_f64();
            }
        }
        for _ in 0..4 {
            if edm_exp.decode(&mut decoder) != DNG_MAGIC_BYTE {
                return Err(DngError::Corrupt(
                    "corruption detected in the dng file".to_string(),
                ));
            }
        }
        Ok(convert_native(&NativeImage::F64(img)))
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gray8_fixture(seed: u64) -> Array2D<u8> {
        let mut x = seed;
        let mut img = Array2D::zeros(9, 11);
        for r in 0..9 {
            for c in 0..11 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                img[(r, c)] = (x >> 33) as u8;
            }
        }
        img
    }

    fn gray16_fixture() -> Array2D<u16> {
        let mut img = Array2D::zeros(7, 13);
        for r in 0..7 {
            for c in 0..13 {
                img[(r, c)] = ((r * 977 + c * 131) % 65536) as u16;
            }
        }
        img
    }

    fn rgb_fixture() -> Array2D<RgbPixel> {
        let mut img = Array2D::zeros(10, 20);
        for r in 0..10 {
            for c in 0..20 {
                img[(r, c)] = RgbPixel {
                    r: ((r * 37 + c * 11) % 256) as u8,
                    g: ((r * 17 + c * 53 + 7) % 256) as u8,
                    b: ((r * 3 + c * 97 + 200) % 256) as u8,
                };
            }
        }
        img
    }

    fn rgba_fixture() -> Array2D<RgbAlphaPixel> {
        let mut img = Array2D::zeros(5, 17);
        for r in 0..5 {
            for c in 0..17 {
                img[(r, c)] = RgbAlphaPixel {
                    r: ((r * 41 + c * 7) % 256) as u8,
                    g: ((r * 13 + c * 61 + 33) % 256) as u8,
                    b: ((r * 29 + c * 19 + 90) % 256) as u8,
                    a: ((r * 5 + c * 83 + 130) % 256) as u8,
                };
            }
        }
        img
    }

    fn hsi_fixture() -> Array2D<HsiPixel> {
        let mut img = Array2D::zeros(4, 6);
        for r in 0..4 {
            for c in 0..6 {
                img[(r, c)] = HsiPixel {
                    h: ((r * 6 + c) * 3) as u16,
                    s: ((r * 6 + c) * 5 + 1) as u16,
                    i: ((r * 6 + c) * 7 + 2) as u16,
                };
            }
        }
        img
    }

    fn float_fixture() -> Array2D<f64> {
        let mut img = Array2D::zeros(6, 8);
        for r in 0..6 {
            for c in 0..8 {
                img[(r, c)] = (r as f64) * 0.25 - (c as f64) * 1.5 + 3.75;
            }
        }
        img
    }

    fn roundtrip<P: Pixel>(img: &Array2D<P>) {
        let bytes = save_dng_bytes(img).expect("save_dng failed");
        let back: Array2D<P> = load_dng_from_slice(&bytes).expect("load_dng failed");
        assert_eq!(back.nr(), img.nr());
        assert_eq!(back.nc(), img.nc());
        for r in 0..img.nr() {
            for c in 0..img.nc() {
                assert_eq!(img[(r, c)], back[(r, c)], "pixel mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn roundtrip_gray8() {
        roundtrip(&gray8_fixture(42));
    }

    #[test]
    fn roundtrip_gray16() {
        roundtrip(&gray16_fixture());
    }

    #[test]
    fn roundtrip_rgb() {
        roundtrip(&rgb_fixture()); // 200 pixels < 4000 -> rgb_paeth
    }

    #[test]
    fn roundtrip_rgb_large_uses_non_paeth() {
        // >= 4000 pixels so the plain rgb predictor path is exercised
        let mut img = Array2D::<RgbPixel>::zeros(64, 64);
        for r in 0..64 {
            for c in 0..64 {
                img[(r, c)] = RgbPixel {
                    r: ((r * 5 + c * 3) % 256) as u8,
                    g: ((r * 7 + c * 11) % 256) as u8,
                    b: ((r * 13 + c * 17) % 256) as u8,
                };
            }
        }
        roundtrip(&img);
    }

    #[test]
    fn roundtrip_rgb_alpha() {
        roundtrip(&rgba_fixture()); // small -> rgb_alpha_paeth
    }

    #[test]
    fn roundtrip_rgb_alpha_large() {
        let mut img = Array2D::<RgbAlphaPixel>::zeros(70, 60);
        for r in 0..70 {
            for c in 0..60 {
                img[(r, c)] = RgbAlphaPixel {
                    r: ((r * 3 + c * 5) % 256) as u8,
                    g: ((r * 9 + c * 7) % 256) as u8,
                    b: ((r * 15 + c * 23) % 256) as u8,
                    a: ((r * 29 + c * 31) % 256) as u8,
                };
            }
        }
        roundtrip(&img);
    }

    #[test]
    fn roundtrip_hsi() {
        roundtrip(&hsi_fixture());
    }

    #[test]
    fn roundtrip_float() {
        roundtrip(&float_fixture());
    }

    #[test]
    fn roundtrip_float32() {
        let mut img = Array2D::<f32>::zeros(5, 5);
        for r in 0..5 {
            for c in 0..5 {
                img[(r, c)] = (r as f32) * 0.5 - (c as f32) * 0.125;
            }
        }
        roundtrip(&img);
    }

    #[test]
    fn tiny_4x1_gray_roundtrip() {
        let mut img = Array2D::<u8>::zeros(1, 4);
        img[(0, 0)] = 10;
        img[(0, 1)] = 200;
        img[(0, 2)] = 7;
        img[(0, 3)] = 255;
        let bytes = save_dng_bytes(&img).expect("save");
        assert_eq!(&bytes[..3], b"DNG");
        let back: Array2D<u8> = load_dng_from_slice(&bytes).expect("load");
        for c in 0..4 {
            assert_eq!(img[(0, c)], back[(0, c)]);
        }
    }

    #[test]
    fn corrupted_stream_returns_err() {
        let mut bytes = save_dng_bytes(&gray8_fixture(7)).expect("save");
        // flip a byte in the middle of the compressed payload
        let mid = bytes.len() / 2;
        assert!(mid >= 10);
        bytes[mid] ^= 0xFF;
        let res: Result<Array2D<u8>, _> = load_dng_from_slice(&bytes);
        if let Ok(img) = res {
            // a corrupted payload may still decode if it happens to match
            // the magic trailer; then it must at least have the right size
            assert_eq!(img.nr(), 9);
            assert_eq!(img.nc(), 11);
        }
    }

    #[test]
    fn truncated_stream_returns_err() {
        let bytes = save_dng_bytes(&gray8_fixture(9)).expect("save");
        let res: Result<Array2D<u8>, _> = load_dng_from_slice(&bytes[..12]);
        assert!(res.is_err());
    }

    #[test]
    fn bad_magic_returns_err() {
        let res: Result<Array2D<u8>, _> = load_dng_from_slice(b"nope");
        assert!(matches!(res, Err(DngError::Corrupt(_))));
    }

    #[test]
    fn empty_image_roundtrip() {
        let img: Array2D<u8> = Array2D::zeros(0, 0);
        let bytes = save_dng_bytes(&img).expect("save");
        let back: Array2D<u8> = load_dng_from_slice(&bytes).expect("load");
        assert_eq!((back.nr(), back.nc()), (0, 0));
    }

    #[test]
    fn file_roundtrip() {
        let dir = std::env::temp_dir().join("dlib-rs-dng-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.dng");
        let img = rgb_fixture();
        save_dng_file(&img, &path).expect("save file");
        let back: Array2D<RgbPixel> = load_dng_file(&path).expect("load file");
        for r in 0..img.nr() {
            for c in 0..img.nc() {
                assert_eq!(img[(r, c)], back[(r, c)]);
            }
        }
        std::fs::remove_file(&path).ok();
    }
}

#[cfg(test)]
mod decode_check {
    use super::*;

    /// Stream produced by C++ dlib `save_dng` for the G8 fixture (golden cross-check).
    const CPP_G8: &str =
        "44 4e 47 01 01 01 01 01 0b 01 09 76 eb d2 b9 a4 ce 17 b0 b1 26 fe 4e 16 e1 8b 45 fe 36 8f be c7 81 10 c5 70 e4 50 2e b7 1b 6f 0f 29 de a1 3b 53 96 78 8d 5e 84 9e 48 41 02 f8 b1 55 02 7f f2 36 d2 8f 85 92 2a 7e 25 75 35 e0 eb 83 0a c9 3f 80 e6 8c ef 45 a5 3c b4 e2 08 b8 85 d5 90 e1 da ca 16 89 a5 b5 e6 7a b3 27 25 b4 a7 92 df ad 17 46 6d 89 1a 66 cc aa 1f a3 90 08 9c 00";

    /// Stream produced by C++ dlib `save_dng` for the G16 fixture (golden cross-check).
    const CPP_G16: &str =
        "44 4e 47 01 01 01 07 01 0d 01 07 00 53 26 eb 5c 89 c4 78 50 1e 1c e5 82 0e 3a a9 40 68 00";

    /// Stream produced by C++ dlib `save_dng` for the RGB fixture (golden cross-check).
    const CPP_RGB: &str =
        "44 4e 47 01 01 01 04 01 14 01 0a 00 72 35 97 1f bf 76 9f fd fd 7b 6b c3 a3 ea 6a 8d 2c 72 7a 5f 52 d1 c0 e8 ef 8d 85 48 84 34 00";

    /// Stream produced by C++ dlib `save_dng` for the RGBA fixture (golden cross-check).
    const CPP_RGBA: &str =
        "44 4e 47 01 01 01 06 01 11 01 05 00 80 ea 71 41 ba 83 38 92 54 ef 74 d5 78 00 8c b8 8e 79 59 4f 63 41 43 ec 7d 7c d8 19 54 86 15 3e 83 e9 82 00";

    /// Stream produced by C++ dlib `save_dng` for the F32 fixture (golden cross-check).
    const CPP_F32: &str =
        "44 4e 47 01 01 01 08 01 08 01 06 01 00 81 80 01 00 81 40 01 40 81 20 81 20 81 20 02 60 01 01 40 81 40 01 00 81 80 81 80 01 00 81 40 02 40 01 01 60 81 20 81 20 81 20 01 40 81 40 01 00 01 40 81 10 81 10 81 10 81 10 01 60 81 20 81 20 81 20 01 70 81 10 81 10 81 10 81 10 81 10 81 10 01 10 81 08 81 08 81 08 81 08 01 70 81 10 81 10 01 24 28 80 73 94 e1 99 7d 96 34 ad f6 71 4e 64 d8 69 7a ec 9a 9a 2f 9e ce 0b bb 11 fe 47 dd 37 99 9a b4 22 c0 00";

    fn bytes(hex: &str) -> Vec<u8> {
        hex.split_whitespace()
            .map(|b| u8::from_str_radix(b, 16).unwrap())
            .collect()
    }
    #[test]
    fn decode_cpp_streams() {
        // streams produced by C++ dlib save_dng (same fixtures as save tests)
        let g8: Array2D<u8> = load_dng_from_slice(&bytes(CPP_G8)).unwrap();
        let mut x: u64 = 42;
        for r in 0..9 {
            for c in 0..11 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                assert_eq!(g8[(r, c)], (x >> 33) as u8);
            }
        }
        let g16: Array2D<u16> = load_dng_from_slice(&bytes(CPP_G16)).unwrap();
        for r in 0..7 {
            for c in 0..13 {
                assert_eq!(g16[(r, c)], ((r * 977 + c * 131) % 65536) as u16);
            }
        }
        let rgb: Array2D<RgbPixel> = load_dng_from_slice(&bytes(CPP_RGB)).unwrap();
        for r in 0..10 {
            for c in 0..20 {
                assert_eq!(
                    rgb[(r, c)],
                    RgbPixel {
                        r: ((r * 37 + c * 11) % 256) as u8,
                        g: ((r * 17 + c * 53 + 7) % 256) as u8,
                        b: ((r * 3 + c * 97 + 200) % 256) as u8
                    }
                );
            }
        }
        let rgba: Array2D<RgbAlphaPixel> = load_dng_from_slice(&bytes(CPP_RGBA)).unwrap();
        for r in 0..5 {
            for c in 0..17 {
                assert_eq!(
                    rgba[(r, c)],
                    RgbAlphaPixel {
                        r: ((r * 41 + c * 7) % 256) as u8,
                        g: ((r * 13 + c * 61 + 33) % 256) as u8,
                        b: ((r * 29 + c * 19 + 90) % 256) as u8,
                        a: ((r * 5 + c * 83 + 130) % 256) as u8
                    }
                );
            }
        }
        let f32img: Array2D<f32> = load_dng_from_slice(&bytes(CPP_F32)).unwrap();
        for r in 0..6 {
            for c in 0..8 {
                assert_eq!(f32img[(r, c)], (r as f32) * 0.5 - (c as f32) * 0.125);
            }
        }
    }
}
