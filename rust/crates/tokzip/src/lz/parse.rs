use super::*;

#[derive(Clone, Copy)]
enum Step {
    Literal,
    Match { len: u32, dist_m1: u32 },
    Rep { idx: u8, len: u32 },
}

#[derive(Clone, Copy)]
struct Node {
    price: u32,
    prev: u32,
    step: Step,
    state: u8,
    reps: [u32; 4],
}

#[derive(Clone)]
struct StatePrices {
    literal: u32,
    short_rep: u32,
    reps: [u32; 4],
    history: u32,
    dictionary: u32,
}

impl StatePrices {
    fn new(models: &DocModels, state: usize) -> Self {
        let probs = &models.probs;
        let mat = price_bit(probs[IS_MATCH + state], 1);
        let rep = mat + price_bit(probs[IS_REP + state], 1);
        let explicit = mat + price_bit(probs[IS_REP + state], 0);
        Self {
            literal: price_bit(probs[IS_MATCH + state], 0),
            short_rep: rep
                + price_bit(probs[IS_REP_G0 + state], 0)
                + price_bit(probs[IS_REP0_LONG + state], 0),
            reps: std::array::from_fn(|idx| rep + models.price_rep_choice(state, idx)),
            history: explicit + price_bit(probs[IS_DICT + state], 0),
            dictionary: explicit + price_bit(probs[IS_DICT + state], 1),
        }
    }
}

#[derive(Clone)]
pub(super) struct ParsePrices {
    match_len: [u32; MATCH_MAX + 1],
    rep_len: [u32; MATCH_MAX + 1],
    dict_len: [u32; MATCH_MAX + 1],
    states: [StatePrices; NUM_STATES],
    history: DistPrices,
    dictionary: DistPrices,
}

impl ParsePrices {
    fn new(models: &DocModels) -> Self {
        Self {
            match_len: len_prices(&models.probs, LEN),
            rep_len: len_prices(&models.probs, REP_LEN),
            dict_len: len_prices(&models.probs, DICT_LEN),
            states: std::array::from_fn(|state| StatePrices::new(models, state)),
            history: DistPrices::new(models, HIST_DIST),
            dictionary: DistPrices::new(models, DICT_OFF),
        }
    }
}

struct Search<'a> {
    nodes: &'a mut [Node],
    prices: &'a ParsePrices,
    pairs: &'a mut Vec<(u32, u32)>,
}

const INF: u32 = u32::MAX;

// Freeze prices within a search chunk so speculative paths never adapt the shared models.
pub(super) fn run_encode_optimal(
    rc: &mut Encoder,
    models: &mut DocModels,
    cs: &mut CoderState,
    mf: &mut MatchFinder,
    span: std::ops::Range<usize>,
    inserted: &mut usize,
) {
    const CHUNK: usize = 1024;
    let mut pos = span.start;
    let end = span.end;
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    while pos < end {
        let max_len = MATCH_MAX.min(end - pos);
        let long_rep = (0..4)
            .map(|idx| (idx, mf.len_at(pos, cs.reps[idx] as usize + 1, max_len)))
            .filter(|&(_, len)| len >= NICE_LEN)
            .max_by_key(|&(idx, len)| {
                (
                    len,
                    std::cmp::Reverse(models.price_rep_choice(cs.state, idx)),
                )
            });
        if let Some((idx, len)) = long_rep {
            emit_rep(rc, models, cs, idx, len);
            pos += len;
            continue;
        }
        let end_target = (pos + CHUNK).min(end);
        let max_reach = (end_target + MATCH_MAX).min(end);
        let n = max_reach - pos;
        nodes.clear();
        nodes.resize(
            n + 1,
            Node {
                price: INF,
                prev: 0,
                step: Step::Literal,
                state: 0,
                reps: [0; 4],
            },
        );
        nodes[0].price = 0;
        nodes[0].state = cs.state as u8;
        nodes[0].reps = cs.reps;

        let base = models.base;
        let adapted;
        // At document start the frozen prices depend only on the immutable priors.
        let prices: &ParsePrices = if pos == 0 {
            base.prices
                .get_or_init(|| Box::new(ParsePrices::new(models)))
        } else {
            adapted = ParsePrices::new(models);
            &adapted
        };
        let dlen = mf.dict.bytes.len();

        let mut search = Search {
            nodes: &mut nodes,
            prices,
            pairs: &mut pairs,
        };
        let (target, reach) = search.search(models, mf, pos..end, end_target - pos, inserted);
        search.backtrack(target, reach, &mut steps);
        for step in steps.iter().rev() {
            match *step {
                Step::Literal => {
                    emit_literal(rc, models, cs, &mf.win, pos);
                    pos += 1;
                }
                Step::Match { len, dist_m1 } => {
                    emit_match(rc, models, cs, pos, dlen, len as usize, dist_m1);
                    pos += len as usize;
                }
                Step::Rep { idx, len } => {
                    emit_rep(rc, models, cs, idx as usize, len as usize);
                    pos += len as usize;
                }
            }
        }
    }
}

impl Search<'_> {
    fn search(
        &mut self,
        models: &mut DocModels,
        mf: &mut MatchFinder,
        span: std::ops::Range<usize>,
        target: usize,
        inserted: &mut usize,
    ) -> (usize, usize) {
        for i in 0..target {
            if self.nodes[i].price == INF {
                continue;
            }
            let gpos = span.start + i;
            while *inserted < gpos {
                mf.insert(*inserted);
                *inserted += 1;
            }
            let state = self.nodes[i].state as usize;
            let reps = self.nodes[i].reps;
            let base = self.nodes[i].price;
            let max_len = MATCH_MAX.min(span.end - gpos);

            let literal_base = base + self.prices.states[state].literal;
            if literal_base < self.nodes[i + 1].price {
                let lit_price = literal_base + models.price_literal(&mf.win, gpos, state, &reps);
                relax(
                    self.nodes,
                    i,
                    1,
                    lit_price,
                    Step::Literal,
                    state_after_literal(state) as u8,
                    reps,
                );
            }

            let mut longest = self.relax_reps(i, gpos, max_len, mf);

            self.pairs.clear();
            if longest < NICE_LEN {
                mf.find_pairs(gpos, max_len, self.pairs);
            }
            if let Some(&(plen, _)) = self.pairs.last() {
                longest = longest.max(plen as usize);
            }
            self.relax_matches(i, gpos, mf.dict.bytes.len());
            if longest >= NICE_LEN {
                return (i + longest, i + longest);
            }
        }

        (target, self.nodes.len() - 1)
    }

    fn backtrack(&self, target: usize, reach: usize, steps: &mut Vec<Step>) {
        // Endpoints cover different byte counts. Comparing total prices alone favors
        // stopping early even when a slightly dearer match covers more input cheaply.
        let mut best = target;
        for j in target..=reach {
            if (self.nodes[j].price as u64) * (best as u64)
                < (self.nodes[best].price as u64) * (j as u64)
            {
                best = j;
            }
        }
        steps.clear();
        let mut cur = best;
        while cur != 0 {
            steps.push(self.nodes[cur].step);
            cur = self.nodes[cur].prev as usize;
        }
    }

    fn relax_reps(&mut self, i: usize, gpos: usize, max_len: usize, mf: &MatchFinder) -> usize {
        let state = self.nodes[i].state as usize;
        let reps = self.nodes[i].reps;
        let base = self.nodes[i].price;
        let mut longest = 0usize;
        for idx in 0..4usize {
            let l = mf.len_at(gpos, reps[idx] as usize + 1, max_len);
            longest = longest.max(l);
            if idx == 0 && l >= 1 {
                let price = base + self.prices.states[state].short_rep;
                let st = if state < 7 { 9 } else { 11 };
                relax(
                    self.nodes,
                    i,
                    1,
                    price,
                    Step::Rep { idx: 0, len: 1 },
                    st,
                    reps,
                );
            }
            if l < 2 {
                continue;
            }
            let choice = base + self.prices.states[state].reps[idx];
            let mut new_reps = reps;
            let rep = new_reps[idx];
            new_reps.copy_within(0..idx, 1);
            new_reps[0] = rep;
            let st = state_after_rep(state) as u8;
            // Keep every short-match length: ending earlier can enable a cheaper
            // continuation. Long matches only contribute their full length to bound work.
            let from = if l >= NICE_LEN { l } else { 2 };
            for (len, &length_price) in self
                .prices
                .rep_len
                .iter()
                .enumerate()
                .take(l + 1)
                .skip(from)
            {
                relax(
                    self.nodes,
                    i,
                    len,
                    choice + length_price,
                    Step::Rep {
                        idx: idx as u8,
                        len: len as u32,
                    },
                    st,
                    new_reps,
                );
            }
        }

        longest
    }

    fn relax_matches(&mut self, i: usize, gpos: usize, dlen: usize) {
        let state = self.nodes[i].state as usize;
        let reps = self.nodes[i].reps;
        let base = self.nodes[i].price;
        let hist_bit = self.prices.states[state].history;
        let dict_bit = self.prices.states[state].dictionary;
        let st = state_after_match(state) as u8;
        let mut len_from = 2usize;
        for &(plen, pdist) in self.pairs.iter() {
            let plen = plen as usize;
            let dist_m1 = pdist - 1;
            let new_reps = [dist_m1, reps[0], reps[1], reps[2]];
            let is_dict = pdist as usize > gpos;
            let (prices, value, kind_bit, len_price): (
                &DistPrices,
                u32,
                u32,
                &[u32; MATCH_MAX + 1],
            ) = if is_dict {
                (
                    &self.prices.dictionary,
                    (dlen - (pdist as usize - gpos)) as u32,
                    dict_bit,
                    &self.prices.dict_len,
                )
            } else {
                (
                    &self.prices.history,
                    dist_m1,
                    hist_bit,
                    &self.prices.match_len,
                )
            };
            let mut dist_price = [0u32; NUM_LEN_TO_POS];
            for (ls, p) in dist_price.iter_mut().enumerate() {
                *p = prices.price(ls, value);
            }
            let from = if plen >= NICE_LEN { plen } else { len_from };
            for (len, &length_price) in len_price.iter().enumerate().take(plen + 1).skip(from) {
                let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
                relax(
                    self.nodes,
                    i,
                    len,
                    base + kind_bit + length_price + dist_price[len_state],
                    Step::Match {
                        len: len as u32,
                        dist_m1,
                    },
                    st,
                    new_reps,
                );
            }
            len_from = plen + 1;
        }
    }
}

#[inline]
fn relax(
    nodes: &mut [Node],
    from: usize,
    adv: usize,
    price: u32,
    step: Step,
    state: u8,
    reps: [u32; 4],
) {
    let to = from + adv;
    if to < nodes.len() && price < nodes[to].price {
        nodes[to] = Node {
            price,
            prev: from as u32,
            step,
            state,
            reps,
        };
    }
}
